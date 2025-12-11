import os
import re
import math
import random
import itertools
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import adjusted_rand_score
from transformers import BertConfig, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 1. NETLIST PARSING (Format B)
# ============================================================

SUPPORTED_BIN_OPS = {'&', '|', '^'}
SUPPORTED_UNARY_OPS = {'~'}


def normalize_escaped_ids(verilog_text: str) -> str:
    """Convert \a[0]  →  a_0"""
    def repl(m):
        raw = m.group(1)
        name = raw.replace('[', '_').replace(']', '')
        return name + ' '
    return re.sub(r'\\([^\s]+)\s', repl, verilog_text)


@dataclass
class ExprNode:
    kind: str                # 'LEAF', 'NOT', 'BIN'
    op: Optional[str] = None
    left: Optional['ExprNode'] = None
    right: Optional['ExprNode'] = None
    name: Optional[str] = None


class EPFLNetlist:
    def __init__(self, name: str):
        self.name = name
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.assigns: Dict[str, ExprNode] = {}

    def __repr__(self):
        return f"EPFLNetlist(name={self.name}, inputs={len(self.inputs)}, outputs={len(self.outputs)}, assigns={len(self.assigns)})"


def tokenize_expr(rhs: str) -> List[str]:
    return re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[~&|^()]', rhs)


def parse_expr_tokens(tokens: List[str]) -> ExprNode:
    """Pratt parser for &, |, ^, ~."""
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def consume():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        return t

    def prec(op):
        return {'&': 3, '^': 2, '|': 1}.get(op, 0)

    def parse_primary():
        t = peek()
        if t is None:
            raise ValueError("Unexpected end")
        if t == '(':
            consume()
            node = parse_expr(0)
            if peek() != ')':
                raise ValueError("Missing )")
            consume()
            return node
        if t == '~':
            consume()
            return ExprNode(kind='NOT', op='~', left=parse_primary())
        if re.match(r'[A-Za-z_][A-Za-z0-9_]*', t):
            consume()
            return ExprNode(kind='LEAF', name=t)
        raise ValueError(f"Bad token: {t}")

    def parse_expr(min_prec):
        left = parse_primary()
        while True:
            t = peek()
            if t not in SUPPORTED_BIN_OPS:
                break
            p = prec(t)
            if p < min_prec:
                break
            op = consume()
            right = parse_expr(p + 1)
            left = ExprNode(kind='BIN', op=op, left=left, right=right)
        return left

    return parse_expr(0)


def parse_epfl_verilog(path: str) -> EPFLNetlist:
    with open(path) as f:
        text = f.read()

    text = normalize_escaped_ids(text)
    text = re.sub(r'//.*', '', text)

    net = EPFLNetlist(os.path.basename(path))

    # Inputs
    for m in re.finditer(r'\binput\b([^;]+);', text):
        names = [x.strip() for x in m.group(1).split(',')]
        for n in names:
            n = re.sub(r'\[[^\]]+\]', '', n).strip()
            if n and n not in net.inputs:
                net.inputs.append(n)

    # Outputs
    for m in re.finditer(r'\boutput\b([^;]+);', text):
        names = [x.strip() for x in m.group(1).split(',')]
        for n in names:
            n = re.sub(r'\[[^\]]+\]', '', n).strip()
            if n and n not in net.outputs:
                net.outputs.append(n)

    # Assigns
    for m in re.finditer(r'\bassign\b\s+([^=]+?)=\s*([^;]+);', text):
        lhs = m.group(1).strip().split()[0]
        rhs = m.group(2).strip()
        toks = tokenize_expr(rhs)
        try:
            expr = parse_expr_tokens(toks)
            net.assigns[lhs] = expr
        except Exception as e:
            print(f"[WARN] Could not parse assign: {rhs}")

    print(f"Parsed {net.name}: #inputs={len(net.inputs)}, #outputs={len(net.outputs)}, #assigns={len(net.assigns)}")
    return net


# ============================================================
# 2. *** FIXED SECTION ***  —  Bounded Depth Tree Builder (K=6)
# ============================================================

@dataclass
class Node:
    kind: str     # 'AND', 'OR', 'XOR', 'NOT', 'LEAF'
    left: any = None
    right: any = None
    name: str = None


def build_bounded_tree(root_signal: str, assigns: Dict[str, ExprNode], K=6) -> Node:
    """
    Build a canonical 2-child logic tree for a signal, but ONLY expand up to
    depth K. This completely avoids recursion and exponential blowup.
    """
    queue = [(root_signal, 0)]
    memo = {}

    def make_leaf(sig):
        return Node(kind="LEAF", name=sig)

    while queue:
        sig, depth = queue.pop()

        if sig in memo:
            continue

        # depth limit
        if depth >= K:
            memo[sig] = make_leaf(sig)
            continue

        # No defining assign → PI leaf
        if sig not in assigns:
            memo[sig] = make_leaf(sig)
            continue

        ast = assigns[sig]

        if ast.kind == 'NOT':
            child = ast.left.name
            memo[sig] = Node(kind='NOT', left=child)
            queue.append((child, depth + 1))

        elif ast.kind == 'BIN':
            lhs = ast.left.name
            rhs = ast.right.name
            op = ast.op
            memo[sig] = Node(kind={'&': 'AND', '|': 'OR', '^': 'XOR'}[op],
                             left=lhs, right=rhs)
            queue.append((lhs, depth + 1))
            queue.append((rhs, depth + 1))

        else:
            memo[sig] = make_leaf(sig)

    # Second pass: convert child names → actual Node references
    def link(node: Node):
        if node.kind == 'LEAF':
            return node
        if node.kind == 'NOT':
            return Node(kind='NOT', left=link(memo[node.left]))
        if node.kind in ['AND', 'OR', 'XOR']:
            return Node(kind=node.kind,
                        left=link(memo[node.left]),
                        right=link(memo[node.right]))
        return node

    return link(memo[root_signal])


# ============================================================
# 3. TREE → TOKEN SEQUENCE  (no recursion danger)
# ============================================================

MAX_SEQ_LEN = 512  # per your choice

def tree_to_tokens(node: Node) -> List[str]:
    tokens = []

    def visit(n: Node):
        if n.kind == "LEAF":
            tokens.append("X")
        elif n.kind == "NOT":
            tokens.append("NOT")
            visit(n.left)
        else:  # AND, OR, XOR
            tokens.append(n.kind)
            visit(n.left)
            visit(n.right)

    visit(node)
    return tokens


# ============================================================
# 4. VOCAB ETC.  (No changes)
# ============================================================

SPECIAL_TOKENS = ['[PAD]', '[CLS]', '[SEP]', '[UNK]']


def build_vocab(all_token_seqs):
    vocab = set()
    for seq in all_token_seqs:
        vocab.update(seq)
    vocab_list = SPECIAL_TOKENS + sorted(vocab)
    tok2id = {t: i for i, t in enumerate(vocab_list)}
    id2tok = {i: t for t, i in tok2id.items()}
    print("Vocab size:", len(tok2id))
    return tok2id, id2tok


def encode_tokens(tokens, tok2id, max_len):
    ids = [tok2id['[CLS]']] + [tok2id.get(t, tok2id['[UNK]']) for t in tokens] + [tok2id['[SEP]']]
    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = tok2id['[SEP]']
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(tok2id['[PAD]'])
        mask.append(0)
    return ids, mask


# ============================================================
# 5. STRUCTURAL SIMILARITY / PAIR BUILDER  (unchanged)
# ============================================================

@dataclass
class BitSeq:
    circuit: str
    bit_name: str
    tokens: List[str]
    bigrams: set


def bigram_set(tokens):
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


def jaccard_similarity(s1, s2):
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def build_bit_sequences_for_circuit(netlist: EPFLNetlist) -> List[BitSeq]:
    bit_seqs = []
    for out in netlist.outputs:
        tree = build_bounded_tree(out, netlist.assigns, K=6)
        tokens = tree_to_tokens(tree)
        bit_seqs.append(BitSeq(
            circuit=netlist.name,
            bit_name=out,
            tokens=tokens,
            bigrams=bigram_set(tokens)
        ))
    print(f"{netlist.name}: {len(bit_seqs)} outputs → {len(bit_seqs)} bit sequences")
    return bit_seqs


def build_pairs_for_circuit(bit_seqs, pos_thresh=0.6, neg_thresh=0.3,
                            max_pos=5000, max_neg=5000):
    pos, neg = [], []
    n = len(bit_seqs)
    for i in range(n):
        for j in range(i + 1, n):
            if len(pos) >= max_pos and len(neg) >= max_neg:
                break
            bi, bj = bit_seqs[i], bit_seqs[j]
            sim = jaccard_similarity(bi.bigrams, bj.bigrams)
            if sim >= pos_thresh and len(pos) < max_pos:
                pos.append((bi, bj, 1))
            elif sim <= neg_thresh and len(neg) < max_neg:
                neg.append((bi, bj, 0))
        if len(pos) >= max_pos and len(neg) >= max_neg:
            break

    print(f"{bit_seqs[0].circuit}: {len(pos)} pos, {len(neg)} neg, total {len(pos)+len(neg)}")
    return pos + neg


# ============================================================
# 6. Dataset, Dataloader (unchanged)
# ============================================================

class PairDataset(Dataset):
    def __init__(self, pairs, tok2id, max_len=MAX_SEQ_LEN):
        self.pairs = pairs
        self.tok2id = tok2id
        self.max_len = max_len

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        bi, bj, lab = self.pairs[idx]
        ids1, m1 = encode_tokens(bi.tokens, self.tok2id, self.max_len)
        ids2, m2 = encode_tokens(bj.tokens, self.tok2id, self.max_len)
        return {
            "input_ids_1": torch.tensor(ids1),
            "attention_mask_1": torch.tensor(m1),
            "input_ids_2": torch.tensor(ids2),
            "attention_mask_2": torch.tensor(m2),
            "label": torch.tensor(lab, dtype=torch.float),
            "circuit": bi.circuit
        }


def collate_batch(batch):
    return {
        'input_ids_1': torch.stack([b["input_ids_1"] for b in batch]),
        'attention_mask_1': torch.stack([b["attention_mask_1"] for b in batch]),
        'input_ids_2': torch.stack([b["input_ids_2"] for b in batch]),
        'attention_mask_2': torch.stack([b["attention_mask_2"] for b in batch]),
        'labels': torch.stack([b["label"] for b in batch]),
        'circuits': [b["circuit"] for b in batch]
    }


# ============================================================
# 7. ReBERT model (unchanged)
# ============================================================

class ReBERTPairClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=4):
        super().__init__()
        cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=MAX_SEQ_LEN + 10,
            type_vocab_size=1
        )
        self.bert = BertModel(cfg)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, ids1, mask1, ids2, mask2, labels=None):
        out1 = self.bert(input_ids=ids1, attention_mask=mask1)
        out2 = self.bert(input_ids=ids2, attention_mask=mask2)
        cls1 = out1.last_hidden_state[:, 0]
        cls2 = out2.last_hidden_state[:, 0]
        logits = self.classifier(torch.cat([cls1, cls2], dim=-1)).squeeze(-1)
        if labels is None:
            return logits
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, logits


# ============================================================
# 8. Training Loop (unchanged)
# ============================================================

def train_rebert(model, train_loader, val_loader=None, num_epochs=5, lr=2e-4):
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        tot, batches = 0, 0
        for batch in train_loader:
            ids1 = batch['input_ids_1'].to(device)
            mask1 = batch['attention_mask_1'].to(device)
            ids2 = batch['input_ids_2'].to(device)
            mask2 = batch['attention_mask_2'].to(device)
            labs = batch['labels'].to(device)

            optim.zero_grad()
            loss, logits = model(ids1, mask1, ids2, mask2, labels=labs)
            loss.backward()
            optim.step()

            tot += loss.item()
            batches += 1

        print(f"[Epoch {epoch}] Train loss = {tot/max(1,batches):.4f}")

        if val_loader:
            model.eval()
            vt, vb, corr, total = 0, 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    ids1 = batch['input_ids_1'].to(device)
                    mask1 = batch['attention_mask_1'].to(device)
                    ids2 = batch['input_ids_2'].to(device)
                    mask2 = batch['attention_mask_2'].to(device)
                    labs = batch['labels'].to(device)
                    loss, logits = model(ids1, mask1, ids2, mask2, labels=labs)
                    vt += loss.item()
                    vb += 1
                    preds = (torch.sigmoid(logits) >= 0.5).long()
                    corr += (preds == labs.long()).sum().item()
                    total += labs.numel()
            print(f"         Val loss={vt/max(1,vb):.4f}, Val acc={corr/max(1,total):.4f}")


# ============================================================
# 9. MAIN PIPELINE (unchanged)
# ============================================================

def main():
    NETLIST_DIR = "/content/epfl"
    train_frac = 0.8
    num_epochs = 5
    batch_size = 32

    files = sorted([f for f in os.listdir(NETLIST_DIR) if f.endswith(".v")])
    print("Found Verilog files:", files)

    all_bitseqs = []
    per_circuit = {}

    for fname in files:
        net = parse_epfl_verilog(os.path.join(NETLIST_DIR, fname))
        bitseqs = build_bit_sequences_for_circuit(net)
        all_bitseqs.extend(bitseqs)
        per_circuit[net.name] = bitseqs

    tok2id, id2tok = build_vocab([b.tokens for b in all_bitseqs])

    all_pairs = []
    for cname, bs in per_circuit.items():
        if len(bs) >= 2:
            all_pairs += build_pairs_for_circuit(bs)

    random.shuffle(all_pairs)
    print("Total pairs:", len(all_pairs))

    n_train = int(len(all_pairs) * train_frac)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    train_ds = PairDataset(train_pairs, tok2id, MAX_SEQ_LEN)
    val_ds = PairDataset(val_pairs, tok2id, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = ReBERTPairClassifier(vocab_size=len(tok2id))
    print(model)

    train_rebert(model, train_loader, val_loader, num_epochs)

    os.makedirs("./rebert_epfl_model", exist_ok=True)
    torch.save(model.state_dict(), "./rebert_epfl_model/rebert_epfl.pt")
    with open("./rebert_epfl_model/vocab.txt", "w") as f:
        for i in range(len(tok2id)):
            f.write(id2tok[i] + "\n")

    print("Saved model and vocab.")


if __name__ == "__main__":
    main()