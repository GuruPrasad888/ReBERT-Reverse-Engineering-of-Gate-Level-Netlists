import os
import re
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertConfig, BertModel

from sklearn.metrics import adjusted_rand_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 1. Fixed ReBERT-style vocabulary (as you saw printed)
# ============================================================

SPECIAL_TOKENS = ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]']
STRUCT_TOKENS = [
    'AND', 'BUF', 'CONST', 'CYCLE', 'DEPTH_CUT', 'FF_Q',
    'INV', 'NAND', 'NOR', 'NOT', 'OR', 'PI',
    'UNDEF', 'XNOR', 'XOR'
]

ALL_TOKENS = SPECIAL_TOKENS + STRUCT_TOKENS
tok2id: Dict[str, int] = {t: i for i, t in enumerate(ALL_TOKENS)}
id2tok: Dict[int, str] = {i: t for t, i in tok2id.items()}

PAD_ID = tok2id['[PAD]']
CLS_ID = tok2id['[CLS]']
SEP_ID = tok2id['[SEP]']
UNK_ID = tok2id['[UNK]']

print("Vocab:", tok2id)

# Maximum sequence length for structural cones (counting CLS/SEP)
MAX_SEQ_LEN = 64

# ============================================================
# 2. Netlist Parsing (ISCAS-89 style with dff)
# ============================================================

@dataclass
class Gate:
    gate_type: str        # 'AND', 'OR', 'NAND', 'NOR', 'INV', 'BUF', etc.
    output: str
    inputs: List[str]


@dataclass
class FF:
    name: str
    q: str
    d: str


@dataclass
class Circuit:
    name: str
    pis: List[str]
    pos: List[str]
    const_nets: Set[str]
    clock_nets: Set[str]
    gates: List[Gate]
    ffs: List[FF]
    drivers: Dict[str, str]          # net -> driver kind: 'GATE:<idx>' or 'FF_Q:<idx>' or 'CONST' or 'PI'
    gate_map: Dict[str, Gate]        # output_net -> Gate
    ff_by_q: Dict[str, FF]           # Q net -> FF
    ff_by_d: Dict[str, FF]           # D net -> FF (optional)


def split_modules(verilog_text: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (module_name, port_list_str, body_str) for each module.
    """
    pattern = r'module\s+(\w+)\s*\((.*?)\);\s*(.*?)endmodule'
    modules = re.findall(pattern, verilog_text, flags=re.S | re.I)
    return modules


def parse_port_list(port_str: str) -> List[str]:
    return [p.strip() for p in port_str.split(',') if p.strip()]


def parse_iscas_netlist(path: str) -> Circuit:
    with open(path, 'r') as f:
        text = f.read()

    text = re.sub(r'//.*', '', text)
    text = re.sub(r'^\s*#.*$', '', text, flags=re.M)
    text = re.sub(r'^\s*//\#.*$', '', text, flags=re.M)

    modules = split_modules(text)
    if not modules:
        raise ValueError(f"No modules found in {path}")

    main_name = None
    main_ports = None
    main_body = None
    for name, ports, body in modules:
        if name.lower() not in ['dff', 'ff', 'dffr']:
            main_name, main_ports, main_body = name, ports, body
            break

    if main_name is None:
        raise ValueError(f"No non-dff module found in {path}")

    ports_list = parse_port_list(main_ports)

    # Inputs & outputs from declarations inside body
    pis: List[str] = []
    pos: List[str] = []

    for m in re.finditer(r'\binput\b([^;]+);', main_body, flags=re.I):
        body = m.group(1)
        nets = [x.strip() for x in body.split(',') if x.strip()]
        for n in nets:
            n = re.sub(r'\[[^\]]+\]', '', n).strip()
            if n and n not in pis:
                pis.append(n)

    for m in re.finditer(r'\boutput\b([^;]+);', main_body, flags=re.I):
        body = m.group(1)
        nets = [x.strip() for x in body.split(',') if x.strip()]
        for n in nets:
            n = re.sub(r'\[[^\]]+\]', '', n).strip()
            if n and n not in pos:
                pos.append(n)

    # Try to guess clock / const nets
    const_nets = set(['GND', 'VDD', "1'b0", "1'b1", '1', '0'])
    # treat these as possible clock names
    clock_candidates = set(['CK', 'CLK', 'CLOCK', 'clk', 'ck'])
    clock_nets = set([n for n in pis if n in clock_candidates])

    # Primitive gates & FFs
    gates: List[Gate] = []
    ffs: List[FF] = []

    # Parse FF instances (dff)
    for m in re.finditer(r'\b(dff|DFF)\b\s+(\w+)\s*\(([^;]+)\);', main_body):
        _, inst_name, arg_str = m.groups()
        args = [a.strip() for a in arg_str.split(',') if a.strip()]
        if len(args) != 3:
            # Unexpected, but continue
            continue
        ck, q, d = args
        ff = FF(name=inst_name, q=q, d=d)
        ffs.append(ff)

    gate_pattern = r'\b(and|or|nand|nor|not|buf|BUF)\b\s+(\w+)\s*\(([^;]+)\);'
    for m in re.finditer(gate_pattern, main_body):
        gtype, inst_name, arg_str = m.groups()
        gtype_lower = gtype.lower()
        args = [a.strip() for a in arg_str.split(',') if a.strip()]
        if len(args) < 2:
            continue
        out_net = args[0]
        in_nets = args[1:]
        if gtype_lower == 'and':
            gate_type = 'AND'
        elif gtype_lower == 'or':
            gate_type = 'OR'
        elif gtype_lower == 'nand':
            gate_type = 'NAND'
        elif gtype_lower == 'nor':
            gate_type = 'NOR'
        elif gtype_lower == 'not':
            gate_type = 'INV'
        elif gtype_lower in ['buf', 'BUF']:
            gate_type = 'BUF'
        else:
            gate_type = 'UNDEF'
        gates.append(Gate(gate_type=gate_type, output=out_net, inputs=in_nets))

    # Build driver maps
    drivers: Dict[str, str] = {}
    gate_map: Dict[str, Gate] = {}
    ff_by_q: Dict[str, FF] = {}
    ff_by_d: Dict[str, FF] = {}

    # Gates
    for i, g in enumerate(gates):
        gate_map[g.output] = g
        drivers[g.output] = f'GATE:{i}'

    # FFs
    for j, ff in enumerate(ffs):
        ff_by_q[ff.q] = ff
        ff_by_d[ff.d] = ff
        drivers[ff.q] = f'FF_Q:{j}'

    for n in pis:
        if n in const_nets:
            drivers[n] = 'CONST'
        elif n in clock_nets:
            drivers[n] = 'CONST'
        else:
            drivers[n] = 'PI'

    for c in const_nets:
        if c not in drivers:
            drivers[c] = 'CONST'

    circ = Circuit(
        name=os.path.basename(path),
        pis=pis,
        pos=pos,
        const_nets=const_nets,
        clock_nets=clock_nets,
        gates=gates,
        ffs=ffs,
        drivers=drivers,
        gate_map=gate_map,
        ff_by_q=ff_by_q,
        ff_by_d=ff_by_d
    )

    print(f"Parsed {circ.name}: #PIs={len(circ.pis)}, #POs={len(circ.pos)}, "
          f"#FFs={len(circ.ffs)}, #gates={len(circ.gates)}")
    return circ


# ============================================================
# 3. Structural cone extraction from FF D inputs
# ============================================================

@dataclass
class WordSeq:
    circuit: str
    ff_name: str
    q_net: str
    d_net: str
    tokens: List[str]


def build_cone_tokens(
    circ: Circuit,
    root_net: str,
    max_depth: int = 6
) -> List[str]:
    """
    Build a structural cone rooted at root_net (usually FF.D) with bounded depth.

    Preorder encoding:
      - For gate nodes: [GATE_TYPE, children...]
      - Leaves:
            PI       -> 'PI'
            CONST    -> 'CONST'
            FF_Q     -> 'FF_Q'
            cycles   -> 'CYCLE'
            depth cut-> 'DEPTH_CUT'
            unknown  -> 'UNDEF'
    """
    tokens: List[str] = []
    visited: Set[str] = set()

    def visit(net: str, depth: int):
        if depth >= max_depth:
            tokens.append('DEPTH_CUT')
            return

        if net in visited:
            tokens.append('CYCLE')
            return

        visited.add(net)

        drv = circ.drivers.get(net, None)

        if drv is None:
            # Try to categorize by name
            if net in circ.const_nets:
                tokens.append('CONST')
            elif net in circ.ff_by_q:
                tokens.append('FF_Q')
            elif net in circ.pis and net not in circ.clock_nets:
                tokens.append('PI')
            else:
                tokens.append('UNDEF')
            return

        if drv == 'CONST':
            tokens.append('CONST')
            return
        elif drv == 'PI':
            tokens.append('PI')
            return
        elif drv.startswith('FF_Q:'):
            # Do not cross sequential boundaries
            tokens.append('FF_Q')
            return
        elif drv.startswith('GATE:'):
            idx = int(drv.split(':')[1])
            gate = circ.gates[idx]
            gtok = gate.gate_type if gate.gate_type in STRUCT_TOKENS else 'UNDEF'
            tokens.append(gtok)
            for fin in gate.inputs:
                visit(fin, depth + 1)
            return
        else:
            tokens.append('UNDEF')
            return

    visit(root_net, depth=0)
    return tokens


def build_word_seqs_for_circuit(circ: Circuit, max_depth: int = 6) -> List[WordSeq]:
    word_seqs: List[WordSeq] = []
    for ff in circ.ffs:
        d_net = ff.d
        toks = build_cone_tokens(circ, d_net, max_depth=max_depth)
        ws = WordSeq(
            circuit=circ.name,
            ff_name=ff.name,
            q_net=ff.q,
            d_net=ff.d,
            tokens=toks
        )
        word_seqs.append(ws)
    print(f"{circ.name}: {len(word_seqs)} FFs → {len(word_seqs)} structural cones")
    return word_seqs


# ============================================================
# 4. Token encoding & pair generation
# ============================================================

def encode_tokens_seq(tokens: List[str], max_len: int = MAX_SEQ_LEN) -> Tuple[List[int], List[int]]:
    """
    Encode structural tokens with fixed vocab, add CLS/SEP, pad to max_len.
    """
    ids = [CLS_ID]
    for t in tokens:
        ids.append(tok2id.get(t, UNK_ID))
    ids.append(SEP_ID)

    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = SEP_ID

    attn = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(PAD_ID)
        attn.append(0)

    return ids, attn


def bigrams(tokens: List[str]) -> Set[Tuple[str, str]]:
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


def jaccard(s1: Set, s2: Set) -> float:
    if not s1 and not s2:
        return 1.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


@dataclass
class WordSeqExt:
    circuit: str
    ff_name: str
    q_net: str
    d_net: str
    tokens: List[str]
    bigrams: Set[Tuple[str, str]]


def extend_wordseqs(word_seqs: List[WordSeq]) -> List[WordSeqExt]:
    out: List[WordSeqExt] = []
    for w in word_seqs:
        out.append(WordSeqExt(
            circuit=w.circuit,
            ff_name=w.ff_name,
            q_net=w.q_net,
            d_net=w.d_net,
            tokens=w.tokens,
            bigrams=bigrams(w.tokens)
        ))
    return out


def build_pairs_for_circuit(
    wexts: List[WordSeqExt],
    pos_thresh: float = 0.6,
    neg_thresh: float = 0.3,
    max_pos: int = 5000,
    max_neg: int = 5000
):
    """
    Self-supervised pairs:
      - Same-circuit FFs
      - Jaccard(bigrams) >= pos_thresh → positive
      - Jaccard(bigrams) <= neg_thresh → negative
    """
    pos_pairs = []
    neg_pairs = []
    n = len(wexts)
    for i in range(n):
        for j in range(i + 1, n):
            if len(pos_pairs) >= max_pos and len(neg_pairs) >= max_neg:
                break
            wi = wexts[i]
            wj = wexts[j]
            sim = jaccard(wi.bigrams, wj.bigrams)
            if sim >= pos_thresh and len(pos_pairs) < max_pos:
                pos_pairs.append((wi, wj, 1.0))
            elif sim <= neg_thresh and len(neg_pairs) < max_neg:
                neg_pairs.append((wi, wj, 0.0))
        if len(pos_pairs) >= max_pos and len(neg_pairs) >= max_neg:
            break

    cname = wexts[0].circuit if wexts else '??'
    print(f"{cname}: {len(pos_pairs)} pos, {len(neg_pairs)} neg, total {len(pos_pairs) + len(neg_pairs)}")
    return pos_pairs + neg_pairs


# ============================================================
# 5. Dataset & DataLoader
# ============================================================

class WordPairDataset(Dataset):
    def __init__(self, pairs, max_len: int = MAX_SEQ_LEN):
        self.pairs = pairs
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        wi, wj, label = self.pairs[idx]
        ids1, mask1 = encode_tokens_seq(wi.tokens, self.max_len)
        ids2, mask2 = encode_tokens_seq(wj.tokens, self.max_len)
        return {
            'input_ids_1': torch.tensor(ids1, dtype=torch.long),
            'attention_mask_1': torch.tensor(mask1, dtype=torch.long),
            'input_ids_2': torch.tensor(ids2, dtype=torch.long),
            'attention_mask_2': torch.tensor(mask2, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
            'circuit': wi.circuit
        }


def collate_batch(batch):
    input_ids_1 = torch.stack([b['input_ids_1'] for b in batch], dim=0)
    attention_mask_1 = torch.stack([b['attention_mask_1'] for b in batch], dim=0)
    input_ids_2 = torch.stack([b['input_ids_2'] for b in batch], dim=0)
    attention_mask_2 = torch.stack([b['attention_mask_2'] for b in batch], dim=0)
    labels = torch.stack([b['label'] for b in batch], dim=0)
    circuits = [b['circuit'] for b in batch]
    return {
        'input_ids_1': input_ids_1,
        'attention_mask_1': attention_mask_1,
        'input_ids_2': input_ids_2,
        'attention_mask_2': attention_mask_2,
        'labels': labels,
        'circuits': circuits
    }


# ============================================================
# 6. ReBERT-style BERT encoder + pair classifier
# ============================================================

class ReBERTPairClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 4):
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=MAX_SEQ_LEN + 10,
            type_vocab_size=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.bert = BertModel(config)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self,
                input_ids_1,
                attention_mask_1,
                input_ids_2,
                attention_mask_2,
                labels=None):
        out1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        cls1 = out1.last_hidden_state[:, 0, :]  # [B, H]

        out2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)
        cls2 = out2.last_hidden_state[:, 0, :]

        pair_repr = torch.cat([cls1, cls2], dim=-1)  # [B, 2H]
        logits = self.classifier(pair_repr).squeeze(-1)  # [B]

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return loss, logits
        else:
            return logits


def train_rebert(model,
                 train_loader,
                 val_loader=None,
                 num_epochs: int = 5,
                 lr: float = 2e-4):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids_1, attention_mask_1,
                                 input_ids_2, attention_mask_2,
                                 labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        print(f"[Epoch {epoch}] Train loss = {avg_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids_1 = batch['input_ids_1'].to(device)
                    attention_mask_1 = batch['attention_mask_1'].to(device)
                    input_ids_2 = batch['input_ids_2'].to(device)
                    attention_mask_2 = batch['attention_mask_2'].to(device)
                    labels = batch['labels'].to(device)

                    loss, logits = model(input_ids_1, attention_mask_1,
                                         input_ids_2, attention_mask_2,
                                         labels=labels)
                    val_loss += loss.item()
                    val_batches += 1
                    preds = (torch.sigmoid(logits) >= 0.5).long()
                    correct += (preds == labels.long()).sum().item()
                    total += labels.numel()
            avg_val_loss = val_loss / max(1, val_batches)
            val_acc = correct / max(1, total)
            print(f"         Val loss = {avg_val_loss:.4f}, Val acc = {val_acc:.4f}")


# ============================================================
# 7. Main pipeline
# ============================================================

def main():
    # ---- config ----
    NETLIST_DIR = "/content/iscas_netlists"   #<<-- UPDATE THIS PATH
    train_frac = 0.8
    batch_size = 32
    num_epochs = 5
    # ----------------

    if not os.path.isdir(NETLIST_DIR):
        print(f"[WARN] Directory {NETLIST_DIR} does not exist. Please update NETLIST_DIR.")
        return

    all_files = [f for f in os.listdir(NETLIST_DIR) if f.endswith(".v")]
    all_files.sort()
    print("Found Verilog files:", all_files)

    all_word_seqs: List[WordSeqExt] = []
    per_circuit_wseqs: Dict[str, List[WordSeqExt]] = {}

    for fname in all_files:
        path = os.path.join(NETLIST_DIR, fname)
        circ = parse_iscas_netlist(path)
        wseqs = build_word_seqs_for_circuit(circ, max_depth=6)
        wexts = extend_wordseqs(wseqs)
        all_word_seqs.extend(wexts)
        per_circuit_wseqs[circ.name] = wexts

    # Build pairs
    all_pairs = []
    for cname, wexts in per_circuit_wseqs.items():
        if len(wexts) < 2:
            continue
        pairs = build_pairs_for_circuit(
            wexts,
            pos_thresh=0.6,
            neg_thresh=0.3,
            max_pos=5000,
            max_neg=5000
        )
        all_pairs.extend(pairs)

    random.shuffle(all_pairs)
    print("Total pairs:", len(all_pairs))
    if len(all_pairs) == 0:
        print("No training pairs formed (not enough FFs?). Exiting.")
        return

    n_train = int(len(all_pairs) * train_frac)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    train_ds = WordPairDataset(train_pairs, max_len=MAX_SEQ_LEN)
    val_ds = WordPairDataset(val_pairs, max_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_batch)

    model = ReBERTPairClassifier(vocab_size=len(tok2id),
                                 hidden_size=256,
                                 num_layers=4,
                                 num_heads=4)
    print(model)

    train_rebert(model, train_loader, val_loader,
                 num_epochs=num_epochs, lr=2e-4)

    # Save model
    save_dir = "./rebert_iscas_model"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "rebert_iscas.pt"))
    with open(os.path.join(save_dir, "vocab.txt"), "w") as f:
        for i in range(len(tok2id)):
            f.write(id2tok[i] + "\n")
    print(f"Saved model + vocab to {save_dir}")


if __name__ == "__main__":
    main()