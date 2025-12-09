import os
import re
import math
import random
import itertools
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import adjusted_rand_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

"""## 1. Netlist parsing (ISCAS89)"""

ISCAS_GATE_TYPES = {"and", "nand", "or", "nor", "not"}

gate_type_norm = {
    "and": "AND",
    "nand": "NAND",
    "or": "OR",
    "nor": "NOR",
    "not": "NOT"
}

dff_like_names = {"dff", "dffr", "dff_n", "dff_p"}

def tokenize_verilog_text(text):
    text = re.sub(r"[(),;]", " ", text)
    toks = text.split()
    return toks

def parse_iscas_netlist(path):
    with open(path, "r") as f:
        text = f.read()
    tokens = tokenize_verilog_text(text)
    print(f"{os.path.basename(path)}: {len(tokens)} tokens (first 15): {tokens[:15]}")

    code = re.sub(r"//.*", "", text)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)

    modules = re.findall(r"module\s+([A-Za-z0-9_]+)\s*\((.*?)\);(.*?)endmodule",
                         code, flags=re.S)
    if not modules:
        raise ValueError(f"No modules found in {path}")

    dff_mod_names = set()
    for mname, ports, body in modules:
        lower = mname.lower()
        if lower in dff_like_names or lower.startswith("dff"):
            dff_mod_names.add(mname)

    top_candidates = [(mname, ports, body) for (mname, ports, body) in modules
                      if mname not in dff_mod_names]
    if not top_candidates:
        top_candidates = [modules[-1]]
    top_module = max(top_candidates, key=lambda m: len(m[1]))
    top_name, top_ports_str, top_body = top_module

    raw_ports = [p.strip() for p in top_ports_str.split(",") if p.strip()]
    primary_inputs = []
    primary_outputs = []
    input_decl = re.findall(r"input\s+([^;]+);", top_body)
    output_decl = re.findall(r"output\s+([^;]+);", top_body)
    input_nets = set()
    output_nets = set()
    for decl in input_decl:
        for name in decl.replace("[", " ").replace("]", " ").replace(":", " ").split(","):
            name = name.strip()
            if name:
                input_nets.add(name)
    for decl in output_decl:
        for name in decl.replace("[", " ").replace("]", " ").replace(":", " ").split(","):
            name = name.strip()
            if name:
                output_nets.add(name)
    for p in raw_ports:
        base = p.split("[")[0].strip()
        if base in input_nets:
            primary_inputs.append(p)
        elif base in output_nets:
            primary_outputs.append(p)
        else:
            primary_outputs.append(p)

    gates = []
    ff_bits = []

    statements = [s.strip() for s in top_body.split(";") if s.strip()]
    inst_re = re.compile(r"^([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\((.*)\)$")

    for stmt in statements:
        m = inst_re.match(stmt)
        if not m:
            continue
        cell_type, inst_name, pinlist = m.groups()
        pinlist = pinlist.strip()
        pins = [p.strip() for p in pinlist.split(",") if p.strip()]
        if not pins:
            continue

        lower_type = cell_type.lower()
        if cell_type in dff_mod_names or lower_type in dff_like_names or lower_type.startswith("dff"):
            if len(pins) >= 2:
                qnet = pins[1]
                ff_bits.append(qnet)
            continue

        gtype = lower_type
        if gtype not in ISCAS_GATE_TYPES:
            continue
        out_net = pins[0]
        in_nets = pins[1:]
        gates.append((out_net, gate_type_norm[gtype], in_nets))

    bits = list(ff_bits)
    if not bits:
        po_set = set(primary_outputs)
        po_drivers = set()
        for out, gtype, ins in gates:
            if out in po_set:
                po_drivers.add(out)
        bits = list(po_drivers)

    print(f"{os.path.basename(path)}: {len(gates)} gates, {len(bits)} bits ({len(ff_bits)} FF bits, {len(bits)-len(ff_bits)} fallback bits)")

    return {
        "name": os.path.basename(path),
        "gates": gates,
        "ff_bits": ff_bits,
        "bits": bits,
        "primary_inputs": primary_inputs,
        "primary_outputs": primary_outputs
    }

"""## 2. Build DAG and Binary Fan-in Trees"""

class CircuitGraph:
    def __init__(self, circ_dict):
        self.name = circ_dict["name"]
        self.gates = circ_dict["gates"]
        self.bits = circ_dict["bits"]
        self.ff_bits = set(circ_dict["ff_bits"])
        self.primary_inputs = set(circ_dict["primary_inputs"])
        self.primary_outputs = set(circ_dict["primary_outputs"])

        self.net_drivers = {}
        self.net_fanins = defaultdict(list)
        self.all_nets = set()

        for out, gtype, ins in self.gates:
            self.net_drivers[out] = (gtype, ins)
            self.net_fanins[out].extend(ins)
            self.all_nets.add(out)
            self.all_nets.update(ins)

        self._build_topological_order()

    def _build_topological_order(self):
        indeg = defaultdict(int)
        adj = defaultdict(list)

        for out, (gtype, ins) in self.net_drivers.items():
            for src in ins:
                if src in self.net_drivers:
                    adj[src].append(out)
                    indeg[out] += 1

        queue = [n for n in self.all_nets if indeg[n] == 0]
        topo = []
        while queue:
            n = queue.pop()
            topo.append(n)
            for nxt in adj[n]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)

        self.topo_order = topo

    def build_binary_tree_for_bit(self, bit_net, max_depth=64):
        cache = {}

        def build_for_net(net, depth):
            if depth > max_depth:
                return {"type": "X", "left": None, "right": None}
            if net in cache:
                return cache[net]
            if net not in self.net_drivers:
                node = {"type": "X", "left": None, "right": None}
                cache[net] = node
                return node

            gtype, ins = self.net_drivers[net]
            if not ins:
                node = {"type": "X", "left": None, "right": None}
                cache[net] = node
                return node

            def chain_inputs(inputs, depth):
                if len(inputs) == 1:
                    return build_for_net(inputs[0], depth+1)
                left = build_for_net(inputs[0], depth+1)
                right = chain_inputs(inputs[1:], depth+1)
                return {"type": gtype, "left": left, "right": right}

            node = chain_inputs(ins, depth)
            cache[net] = node
            return node

        root = build_for_net(bit_net, 0)
        return root

    def tree_to_tokens(self, node, tokens):
        if node is None:
            return
        tokens.append(node["type"])
        self.tree_to_tokens(node["left"], tokens)
        self.tree_to_tokens(node["right"], tokens)


root_dir = "/iscas"
if not os.path.isdir(root_dir):
    os.makedirs(root_dir, exist_ok=True)

netlist_files = [f for f in os.listdir(root_dir) if f.endswith(".v")]
netlist_files = sorted(netlist_files)

print("Found netlists:")
for f in netlist_files:
    print("   ", f)

circuits = {}
for fname in netlist_files:
    path = os.path.join(root_dir, fname)
    circ_dict = parse_iscas_netlist(path)
    circuits[fname] = CircuitGraph(circ_dict)

if netlist_files:
    first_cname = netlist_files[0]
    cg = circuits[first_cname]
    for bit in cg.bits[:5]:
        root = cg.build_binary_tree_for_bit(bit)
        toks = []
        cg.tree_to_tokens(root, toks)
        print(" bit:", bit, "root_net:", bit, "-> tokens (first 15):", toks[:15], "len=", len(toks))
else:
    print("No netlists yet. Please add ISCAS89 .v files to /iscas.")

"""## 3. Functional Simulation and Signatures"""

def simulate_circuit(cg, num_vecs=1024, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    bits = cg.bits
    topo = cg.topo_order
    net_val = {n: 0 for n in cg.all_nets}

    bit_sigs = {b: np.zeros(num_vecs, dtype=np.int8) for b in bits}

    driven = set(cg.net_drivers.keys())
    pi_nets = set()
    for out, (gtype, ins) in cg.net_drivers.items():
        for src in ins:
            if src not in driven:
                pi_nets.add(src)

    for t in range(num_vecs):
        for pi in pi_nets:
            net_val[pi] = random.randint(0, 1)

        for net in topo:
            if net not in cg.net_drivers:
                continue
            gtype, ins = cg.net_drivers[net]
            in_vals = [net_val.get(src, 0) for src in ins]

            if gtype == "AND":
                val = int(all(in_vals))
            elif gtype == "NAND":
                val = int(not all(in_vals))
            elif gtype == "OR":
                val = int(any(in_vals))
            elif gtype == "NOR":
                val = int(not any(in_vals))
            elif gtype == "NOT":
                val = int(not in_vals[0]) if in_vals else 1
            else:
                val = 0

            net_val[net] = val

        for b in bits:
            bit_sigs[b][t] = net_val.get(b, 0)

    return bit_sigs


NUM_SIM_VECS = 1024

print("\nBuilding functional word groups (per circuit) via simulation...")
functional_words = {}
functional_bit_to_word = {}

for cname, cg in circuits.items():
    print(f"   Simulating {cname} ...")
    if not cg.bits:
        functional_words[cname] = []
        functional_bit_to_word[cname] = {}
        print("      0 bits → 0 functional words")
        continue

    bit_sigs = simulate_circuit(cg, num_vecs=NUM_SIM_VECS, seed=0)
    bits = list(cg.bits)
    n = len(bits)
    sigs = [bit_sigs[b].astype(np.bool_) for b in bits]

    eps = min(0.15, max(0.03, 20.0 / n))

    adj = [[] for _ in range(n)]
    for i in range(n):
        si = sigs[i]
        for j in range(i+1, n):
            sj = sigs[j]
            mismatches = np.count_nonzero(si ^ sj)
            dist = mismatches / float(NUM_SIM_VECS)
            if dist <= eps:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        groups.append([bits[k] for k in comp])

    bit_to_word = {}
    for wid, group in enumerate(groups):
        for b in group:
            bit_to_word[b] = wid

    functional_words[cname] = groups
    functional_bit_to_word[cname] = bit_to_word
    print(f"      {len(bits)} bits grouped into {len(groups)} functional words (eps={eps:.4f})")

"""## 4. Structural Similarity and Hybrid Word Grouping"""

def build_structural_signature(cg, bit, max_depth=64):
    root = cg.build_binary_tree_for_bit(bit, max_depth=max_depth)
    toks = []
    cg.tree_to_tokens(root, toks)
    counts = Counter(t for t in toks)
    return counts, toks

def jaccard_from_counts(c1, c2):
    keys = set(c1.keys()) | set(c2.keys())
    if not keys:
        return 0.0
    inter = 0.0
    union = 0.0
    for k in keys:
        v1 = c1.get(k, 0)
        v2 = c2.get(k, 0)
        inter += min(v1, v2)
        union += max(v1, v2)
    if union == 0.0:
        return 0.0
    return inter / union

print("\nBuilding hybrid (functional + structural) word groups (per circuit)...")
hybrid_words = {}
hybrid_bit_to_word = {}

for cname, cg in circuits.items():
    bits = list(cg.bits)
    n = len(bits)
    if n == 0:
        hybrid_words[cname] = []
        hybrid_bit_to_word[cname] = {}
        print(f"{cname}: 0 bits → 0 hybrid words")
        continue

    struct_counts = {}
    for b in bits:
        cnts, toks = build_structural_signature(cg, b)
        struct_counts[b] = cnts

    func_groups = functional_words[cname]
    tau_struct = 0.3

    index_of = {b: i for i, b in enumerate(bits)}
    adj = [[] for _ in range(n)]

    for group in func_groups:
        if len(group) <= 1:
            continue
        for b1, b2 in itertools.combinations(group, 2):
            i = index_of[b1]
            j = index_of[b2]
            s1 = struct_counts[b1]
            s2 = struct_counts[b2]
            js = jaccard_from_counts(s1, s2)
            if js >= tau_struct:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        groups.append([bits[k] for k in comp])

    bit_to_word = {}
    for wid, group in enumerate(groups):
        for b in group:
            bit_to_word[b] = wid

    hybrid_words[cname] = groups
    hybrid_bit_to_word[cname] = bit_to_word

    print(f"{cname}: {len(bits)} bits → {len(groups)} hybrid words")

"""## 5. Build Bit-Pair Dataset from Hybrid Words"""

MAX_PAIRS_PER_CIRCUIT = 5000

def build_pairs_for_circuit(cname, cg, words, bit_to_word):
    bits = list(cg.bits)
    n = len(bits)
    if n < 2 or len(words) == 0:
        return []

    word_to_bits = [list(w) for w in words]
    bit2w = bit_to_word

    pos_pairs = []
    for wid, group in enumerate(word_to_bits):
        if len(group) < 2:
            continue
        for b1, b2 in itertools.combinations(group, 2):
            pos_pairs.append((b1, b2, 1))

    neg_pairs = []
    bits_by_word = defaultdict(list)
    for b in bits:
        if b in bit2w:
            bits_by_word[bit2w[b]].append(b)

    word_ids = [wid for wid, g in enumerate(word_to_bits) if g]
    if len(word_ids) > 1:
        while len(neg_pairs) < len(pos_pairs) and len(neg_pairs) < MAX_PAIRS_PER_CIRCUIT:
            w1, w2 = random.sample(word_ids, 2)
            b1 = random.choice(bits_by_word[w1])
            b2 = random.choice(bits_by_word[w2])
            neg_pairs.append((b1, b2, 0))

    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)
    if len(all_pairs) > MAX_PAIRS_PER_CIRCUIT:
        all_pairs = all_pairs[:MAX_PAIRS_PER_CIRCUIT]

    return all_pairs


print("\nBuilding bit-pair datasets with hybrid labels...")

circuit_pairs = {}
token_to_id = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2}

def add_tokens(toks):
    for t in toks:
        if t not in token_to_id:
            token_to_id[t] = len(token_to_id)

for cname, cg in circuits.items():
    words = hybrid_words[cname]
    bit_to_word = hybrid_bit_to_word[cname]
    pairs = build_pairs_for_circuit(cname, cg, words, bit_to_word)
    circuit_pairs[cname] = pairs
    print(f"{cname}: {len(pairs)} pairs")

    for b in cg.bits:
        root = cg.build_binary_tree_for_bit(b)
        toks = []
        cg.tree_to_tokens(root, toks)
        add_tokens(toks)

print("\nVocab size:", len(token_to_id), "tokens:", list(token_to_id.keys()))

"""## 6. ReBERT Dataset and Model"""

class BitPairDataset(Dataset):
    def __init__(self, cg, pairs, max_depth=64, max_len=128):
        self.cg = cg
        self.pairs = pairs
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def _bit_to_ids(self, bit):
        root = self.cg.build_binary_tree_for_bit(bit)
        toks = []
        self.cg.tree_to_tokens(root, toks)
        toks = toks[: self.max_len - 1]
        ids = [token_to_id.get(t, token_to_id["[UNK]"]) for t in toks]
        ids.append(token_to_id["[SEP]"])
        return ids

    def __getitem__(self, idx):
        b1, b2, y = self.pairs[idx]
        ids1 = self._bit_to_ids(b1)
        ids2 = self._bit_to_ids(b2)
        input_ids = ids1 + ids2
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
        attn_mask = [1] * len(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), torch.tensor(y, dtype=torch.float32)


def collate_batch(batch):
    input_ids, attn_masks, labels = zip(*batch)
    max_len = max(x.size(0) for x in input_ids)
    pad_id = token_to_id["[PAD]"]
    batch_ids = []
    batch_mask = []
    for ids, mask in zip(input_ids, attn_masks):
        pad_len = max_len - ids.size(0)
        batch_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        batch_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))
    return torch.stack(batch_ids), torch.stack(batch_mask), torch.stack(labels)


class ReBERT(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, 1)

    def forward(self, input_ids, attn_mask):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        key_padding_mask = ~attn_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        mask = attn_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        logits = self.cls(x).squeeze(-1)
        return logits


def make_dataloader_for_circuit(cname, batch_size=64, shuffle=True):
    cg = circuits[cname]
    pairs = circuit_pairs[cname]
    if len(pairs) == 0:
        return None
    ds = BitPairDataset(cg, pairs, max_len=128)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)

"""## 7. Training Loop and Leave-One-Out Experiment"""

def train_one_epoch(model, loader, opt):
    model.train()
    total_loss = 0.0
    total = 0
    for input_ids, attn_mask, labels in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        logits = model(input_ids, attn_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / max(total, 1)


def eval_model(model, loader):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for input_ids, attn_mask, labels in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attn_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
    return total_loss / max(total, 1), correct / max(total, 1)


def run_loo_experiment(num_epochs=5, batch_size=64, lr=1e-3):
    results = []
    usable = [c for c in circuits.keys() if circuit_pairs[c]]
    print("Circuits with usable pairs:", usable)
    for test_circ in usable:
        print(f"\n=== Test circuit: {test_circ} ===")
        orig_pairs = circuit_pairs[test_circ]
        circuit_pairs[test_circ] = []

        train_loaders = []
        for cname in circuits.keys():
            if cname == test_circ:
                continue
            dl = make_dataloader_for_circuit(cname, batch_size=batch_size, shuffle=True)
            if dl is not None:
                train_loaders.append(dl)

        circuit_pairs[test_circ] = orig_pairs
        test_loader = make_dataloader_for_circuit(test_circ, batch_size=batch_size, shuffle=False)
        if not train_loaders or test_loader is None:
            print("  Skipping (no train or test data).")
            continue

        model = ReBERT(vocab_size=len(token_to_id)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, num_epochs+1):
            total_loss = 0.0
            total_count = 0
            for dl in train_loaders:
                loss = train_one_epoch(model, dl, opt)
                total_loss += loss
                total_count += 1
            avg_train_loss = total_loss / max(total_count, 1)
            test_loss, test_acc = eval_model(model, test_loader)
            print(f"   Epoch {epoch:02d}: train_loss~{avg_train_loss:.4f}  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

        results.append({"circuit": test_circ, "test_loss": test_loss, "test_acc": test_acc})

    return results


print("\nRunning LOO demo")
demo_results = run_loo_experiment(num_epochs=5, batch_size=64, lr=1e-3)
print("Demo results:", demo_results)

"""## 8. ARI Demo (Hybrid Words as Ground Truth)"""

def compute_ari_for_circuit(cname, model=None):
    cg = circuits[cname]
    words = hybrid_words[cname]
    bit_to_word = hybrid_bit_to_word[cname]
    bits = list(cg.bits)
    if len(bits) < 2:
        print(f"{cname}: not enough bits for ARI.")
        return None

    pairs = circuit_pairs[cname]
    if not pairs:
        print(f"{cname}: no pairs, ARI skipped.")
        return None

    if model is None:
        model = ReBERT(vocab_size=len(token_to_id)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        dl = make_dataloader_for_circuit(cname, batch_size=64, shuffle=True)
        print(f"  Training a local model for {cname}...")
        for epoch in range(2):
            loss = train_one_epoch(model, dl, opt)
            print(f"   Local epoch {epoch+1}: loss={loss:.4f}")

    model.eval()

    embeddings = {}
    with torch.no_grad():
        for b in bits:
            root = cg.build_binary_tree_for_bit(b)
            toks = []
            cg.tree_to_tokens(root, toks)
            toks = toks[:127]
            ids = [token_to_id.get(t, token_to_id["[UNK]"]) for t in toks]
            ids.append(token_to_id["[SEP]"])
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            attn_mask = torch.ones_like(input_ids).to(device)
            B, L = input_ids.shape
            pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
            x = model.token_emb(input_ids) + model.pos_emb(pos)
            key_padding_mask = ~attn_mask.bool()
            x = model.encoder(x, src_key_padding_mask=key_padding_mask)
            mask = attn_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            embeddings[b] = x.squeeze(0).cpu().numpy()

    group_ids = sorted(set(bit_to_word[b] for b in bits))
    centroids = {}
    for gid in group_ids:
        vecs = [embeddings[b] for b in bits if bit_to_word[b] == gid]
        centroids[gid] = np.mean(vecs, axis=0)

    pred_labels = []
    true_labels = []
    for b in bits:
        true_labels.append(bit_to_word[b])
        v = embeddings[b]
        best_gid = None
        best_dist = None
        for gid, c in centroids.items():
            d = np.linalg.norm(v - c)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_gid = gid
        pred_labels.append(best_gid)

    ari = adjusted_rand_score(true_labels, pred_labels)
    return ari


if netlist_files:
    demo_circ = netlist_files[0]
    print("\nComputing ARI (functional+structural GT) on demo circuit:", demo_circ)
    ari = compute_ari_for_circuit(demo_circ)
    print("  ARI (hybrid GT):", ari)
else:
    print("\nNo circuits available for ARI demo.")