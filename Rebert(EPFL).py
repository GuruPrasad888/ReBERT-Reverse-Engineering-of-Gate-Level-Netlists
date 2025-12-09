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

# ============================================================
# 1. AST for boolean expressions (~, &, |, parentheses)
# ============================================================

class Var:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"Var({self.name})"

class Not:
    def __init__(self, child):
        self.child = child
    def __repr__(self):
        return f"Not({self.child})"

class And:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"And({self.left}, {self.right})"

class Or:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Or({self.left}, {self.right})"


def tokenize_expr(s: str):
    """
    Tokenize expression with ~, &, |, parentheses, identifiers.
    """
    for ch in "~&|()":
        s = s.replace(ch, f" {ch} ")
    toks = [t for t in s.split() if t]
    return toks


def parse_expr(tokens):
    """
    Recursive-descent parser with precedence:
        ~ (highest), & (middle), | (lowest)
    Grammar:
        E := T ('|' T)*
        T := F ('&' F)*
        F := '~' F | '(' E ')' | IDENT
    """
    tokens = list(tokens)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def consume(expected=None):
        nonlocal pos
        tok = peek()
        if expected is not None and tok != expected:
            raise SyntaxError(f"expected {expected}, got {tok}")
        pos += 1
        return tok

    def parse_F():
        tok = peek()
        if tok == "~":
            consume("~")
            return Not(parse_F())
        elif tok == "(":
            consume("(")
            node = parse_E()
            if peek() != ")":
                raise SyntaxError("missing ')'")
            consume(")")
            return node
        elif tok is None:
            raise SyntaxError("unexpected end of expression")
        else:
            consume()
            return Var(tok)

    def parse_T():
        node = parse_F()
        while peek() == "&":
            consume("&")
            right = parse_F()
            node = And(node, right)
        return node

    def parse_E():
        node = parse_T()
        while peek() == "|":
            consume("|")
            right = parse_T()
            node = Or(node, right)
        return node

    ast = parse_E()
    if pos != len(tokens):
        raise SyntaxError("extra tokens at end of expression: " + str(tokens[pos:]))
    return ast


def collect_vars(node, acc=None):
    if acc is None:
        acc = set()
    if isinstance(node, Var):
        acc.add(node.name)
    elif isinstance(node, Not):
        collect_vars(node.child, acc)
    elif isinstance(node, (And, Or)):
        collect_vars(node.left, acc)
        collect_vars(node.right, acc)
    return acc


def eval_expr(node, env):
    if isinstance(node, Var):
        return env[node.name]
    if isinstance(node, Not):
        return 1 - eval_expr(node.child, env)
    if isinstance(node, And):
        return eval_expr(node.left, env) & eval_expr(node.right, env)
    if isinstance(node, Or):
        return eval_expr(node.left, env) | eval_expr(node.right, env)
    raise TypeError(f"Unknown node type {type(node)}")


# ============================================================
# 2. Parse EPFL-style AIG netlists (assign-only, combinational)
# ============================================================

def normalize_name(name: str) -> str:
    """
    Keep escaped identifiers like '\\a[0]' as a single token.
    Trim whitespace and trailing commas.
    """
    name = name.strip()
    if name.endswith(","):
        name = name[:-1].strip()
    return name


def parse_epfl_netlist(path: str):

    with open(path, "r") as f:
        text = f.read()

    # Strip comments
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)

    m = re.search(r"module\s+([A-Za-z0-9_\\]+)\s*\((.*?)\)\s*;(.*)endmodule",
                  text, flags=re.S)
    if not m:
        raise ValueError(f"Cannot find module header in {path}")
    mod_name, ports_str, body = m.groups()

    # Port list (may include escaped names like \a[0] )
    raw_ports = [normalize_name(p) for p in ports_str.split(",") if p.strip()]
    port_set = set(raw_ports)

    # Explicit input/output declarations (if present)
    def parse_decl_list(pattern):
        decls = set()
        for decl in re.findall(pattern, body):
            # decl may contain multiple names and vector ranges; keep raw tokens
            parts = [p.strip() for p in decl.split(",") if p.strip()]
            for p in parts:
                # Example: \a[0]
                decls.add(normalize_name(p))
        return decls

    input_nets = parse_decl_list(r"input\s+([^;]+);")
    output_nets = parse_decl_list(r"output\s+([^;]+);")

    # Parse assign statements
    assigns = {}
    for lhs, rhs in re.findall(r"assign\s+(.+?)\s*=\s*(.+?);", body):
        lhs = normalize_name(lhs)
        rhs = rhs.strip()
        try:
            expr_ast = parse_expr(tokenize_expr(rhs))
        except SyntaxError as e:
            raise SyntaxError(f"Error parsing expression in {path} for '{lhs} = {rhs}': {e}")
        assigns[lhs] = expr_ast


    if output_nets:
        bits = [b for b in sorted(output_nets) if b in assigns]
    else:
        bits = [p for p in raw_ports if p in assigns]

    # Primary inputs for reference (not strictly needed after expansion)
    if input_nets:
        primary_inputs = sorted(input_nets)
    else:
        # Heuristic: ports that are NOT LHS of assign
        primary_inputs = sorted(p for p in raw_ports if p not in assigns)

    print(f"<Circuit {os.path.basename(path)}: {len(assigns)} assigns, {len(bits)} bits>")

    return {
        "name": os.path.basename(path),
        "assigns": assigns,
        "bits": bits,
        "primary_inputs": primary_inputs,
    }


# ============================================================
# 3. EPFLCircuit: inline assign chains, get tokens, etc.
# ============================================================

MAX_INLINE_DEPTH = 64 # prevent infinite recursion on cycles

class EPFLCircuit:
    def __init__(self, circ_dict):
        self.name = circ_dict["name"]
        self.assigns = circ_dict["assigns"]     # dict: net name -> expr AST
        self.bits   = list(circ_dict["bits"])   # list of output nets used as bits
        self.primary_inputs = set(circ_dict["primary_inputs"])

        # Cache for expanded expressions per net/bit
        self._expanded_cache = {}

        # Precompute expanded expressions for all bits
        for b in self.bits:
            _ = self.get_bit_expr(b)

    def _expand_var(self, name, depth=0):
        if depth > MAX_INLINE_DEPTH:
            return Var(name)
        if name in self._expanded_cache:
            return self._expanded_cache[name]
        if name not in self.assigns:
            node = Var(name)
            self._expanded_cache[name] = node
            return node
        expr = self.assigns[name]
        node = self._expand_expr(expr, depth + 1)
        self._expanded_cache[name] = node
        return node

    def _expand_expr(self, node, depth=0):
        if isinstance(node, Var):
            return self._expand_var(node.name, depth)
        if isinstance(node, Not):
            return Not(self._expand_expr(node.child, depth + 1))
        if isinstance(node, And):
            return And(self._expand_expr(node.left, depth + 1),
                       self._expand_expr(node.right, depth + 1))
        if isinstance(node, Or):
            return Or(self._expand_expr(node.left, depth + 1),
                      self._expand_expr(node.right, depth + 1))
        raise TypeError(f"Unknown node type {type(node)}")

    def get_bit_expr(self, bit):
        """
        Fully expanded expression for output bit, in terms of only leaf Vars
        that are ultimately **primary inputs** (or undriven nets).
        """
        return self._expand_var(bit, depth=0)

    # ---- Structural tokens from expression tree ----
    def expr_to_tokens(self, expr, toks, max_depth=None, depth=0):
        if max_depth is not None and depth > max_depth:
            # Truncate deep subtrees as generic leaf
            toks.append("X")
            return
        if isinstance(expr, Var):
            toks.append("X")
        elif isinstance(expr, Not):
            toks.append("NOT")
            self.expr_to_tokens(expr.child, toks, max_depth, depth + 1)
        elif isinstance(expr, And):
            toks.append("AND")
            self.expr_to_tokens(expr.left, toks, max_depth, depth + 1)
            self.expr_to_tokens(expr.right, toks, max_depth, depth + 1)
        elif isinstance(expr, Or):
            toks.append("OR")
            self.expr_to_tokens(expr.left, toks, max_depth, depth + 1)
            self.expr_to_tokens(expr.right, toks, max_depth, depth + 1)
        else:
            # Unknown node type, treat as leaf
            toks.append("X")

    def bit_to_tokens(self, bit, max_depth=None):
        expr = self.get_bit_expr(bit)
        toks = []
        self.expr_to_tokens(expr, toks, max_depth=max_depth, depth=0)
        return toks


# ============================================================
# 4. Load all EPFL AIG circuits
# ============================================================

root_dir = "/EPFL"

netlist_files = sorted(
    f for f in os.listdir(root_dir)
    if f.endswith(".v")
)

print("Found netlists:")
for f in netlist_files:
    print("   ", f)

circuits = {}
for fname in netlist_files:
    path = os.path.join(root_dir, fname)
    circ_dict = parse_epfl_netlist(path)
    circuits[fname] = EPFLCircuit(circ_dict)

# Show example tokens
print("\nExample bit expressions and tokens from first circuit (if any):")
if netlist_files:
    first = netlist_files[0]
    cg = circuits[first]
    for bit in cg.bits[:3]:
        expr = cg.get_bit_expr(bit)
        toks = cg.bit_to_tokens(bit, max_depth=32)
        print(f" bit: {bit}")
        print("   expr:", expr)
        print("   tokens (first 15):", toks[:15], "len=", len(toks))
else:
    print("No *.v netlists found in", root_dir)

# ============================================================
# 5. Functional simulation (signatures per output bit)
# ============================================================

NUM_SIM_VECS = 256  # moderate – EPFL has few bits per circuit, so this is fine

def simulate_circuit(cg: EPFLCircuit, num_vecs=NUM_SIM_VECS, seed=0):
    """
    Builds random input patterns and evaluates each output bit expression.
    """
    random.seed(seed)
    np.random.seed(seed)

    bits = cg.bits
    if not bits:
        return {}

    # Collect all leaf vars from all bit expressions
    all_vars = set()
    bit_exprs = {}
    for b in bits:
        expr = cg.get_bit_expr(b)
        bit_exprs[b] = expr
        collect_vars(expr, all_vars)

    all_vars = sorted(all_vars)

    bit_sigs = {b: np.zeros(num_vecs, dtype=np.int8) for b in bits}

    for t in range(num_vecs):
        env = {v: random.randint(0, 1) for v in all_vars}
        for b in bits:
            bit_sigs[b][t] = eval_expr(bit_exprs[b], env)

    return bit_sigs


print("\nBuilding functional + hybrid groups (processing circuits one by one)...")

functional_words = {}
functional_bit_to_word = {}
hybrid_words = {}
hybrid_bit_to_word = {}

def build_functional_groups(cg: EPFLCircuit):
    bits = cg.bits
    n = len(bits)
    if n == 0:
        return [], {}

    bit_sigs = simulate_circuit(cg, num_vecs=NUM_SIM_VECS, seed=0)
    sigs = [bit_sigs[b].astype(np.bool_) for b in bits]

    # Adaptive epsilon as in your ISCAS code
    eps = min(0.15, max(0.03, 20.0 / n))

    adj = [[] for _ in range(n)]
    for i in range(n):
        si = sigs[i]
        for j in range(i + 1, n):
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

    return groups, bit_to_word


def build_structural_signature(cg: EPFLCircuit, bit):
    """
    Structural signature = token multiset from the output bit's expanded expression.
    """
    expr = cg.get_bit_expr(bit)
    toks = []
    cg.expr_to_tokens(expr, toks, max_depth=64)
    counts = Counter(toks)
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


# Process circuits one by one (saves RAM vs holding all signatures at once)
for cname, cg in circuits.items():
    print(f"<Circuit {cname}: {len(cg.assigns)} assigns, {len(cg.bits)} bits>")

    if not cg.bits:
        functional_words[cname] = []
        functional_bit_to_word[cname] = {}
        hybrid_words[cname] = []
        hybrid_bit_to_word[cname] = {}
        print(f"\nCircuit {cname}: 0 bits\n   0 bits → 0 groups")
        continue

    print(f"\nCircuit {cname}: {len(cg.bits)} bits")

    # --- Functional groups ---
    func_groups, bit2func = build_functional_groups(cg)
    functional_words[cname] = func_groups
    functional_bit_to_word[cname] = bit2func

    print(f"   Functional: {len(cg.bits)} bits → {len(func_groups)} words")

    # --- Structural signatures for all bits ---
    struct_counts = {}
    for b in cg.bits:
        cnts, toks = build_structural_signature(cg, b)
        struct_counts[b] = cnts

    # --- Hybrid groups: refinement of functional groups with structural Jaccard ---
    tau_struct = 0.3
    bits = list(cg.bits)
    n = len(bits)
    index_of = {b: i for i, b in enumerate(bits)}
    adj = [[] for _ in range(n)]

    for group in func_groups:
        if len(group) <= 1:
            continue
        for b1, b2 in itertools.combinations(group, 2):
            i = index_of[b1]
            j = index_of[b2]
            js = jaccard_from_counts(struct_counts[b1], struct_counts[b2])
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

    print(f"   Hybrid: {len(cg.bits)} bits → {len(groups)} words")

# ============================================================
# 6. Build bit-pair datasets (hybrid labels) & vocab
# ============================================================

MAX_PAIRS_PER_CIRCUIT = 2000  # limit dataset size per circuit

def build_pairs_for_circuit(cname, cg, words, bit_to_word):
    bits = list(cg.bits)
    n = len(bits)
    if n < 2 or len(words) == 0:
        return []

    word_to_bits = [list(w) for w in words]
    bit2w = bit_to_word

    # Positive pairs: bits within same hybrid word
    pos_pairs = []
    for wid, group in enumerate(word_to_bits):
        if len(group) < 2:
            continue
        for b1, b2 in itertools.combinations(group, 2):
            pos_pairs.append((b1, b2, 1))

    # Negative pairs: bits from different words
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

# Build pairs + vocab
for cname, cg in circuits.items():
    words = hybrid_words[cname]
    bit_to_word = hybrid_bit_to_word[cname]
    pairs = build_pairs_for_circuit(cname, cg, words, bit_to_word)
    circuit_pairs[cname] = pairs
    print(f"{cname}: {len(pairs)} pairs")

    for b in cg.bits:
        toks = cg.bit_to_tokens(b, max_depth=64)
        add_tokens(toks)

print("\nVocab size:", len(token_to_id), "tokens:", list(token_to_id.keys()))

# ============================================================
# 7. ReBERT Dataset & Model
# ============================================================

class BitPairDataset(Dataset):
    def __init__(self, cg: EPFLCircuit, pairs, max_len=128):
        self.cg = cg
        self.pairs = pairs
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def _bit_to_ids(self, bit):
        toks = self.cg.bit_to_tokens(bit, max_depth=64)
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
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn_mask, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )

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
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu",
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

# ============================================================
# 8. Training loop & LOO experiment
# ============================================================

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

        for epoch in range(1, num_epochs + 1):
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


print("\nRunning LOO experiment")
demo_results = run_loo_experiment(num_epochs=1, batch_size=64, lr=1e-3)
print("Demo results:", demo_results)

# ============================================================
# 9. ARI demo using hybrid groups as “ground truth”
# ============================================================

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
            toks = cg.bit_to_tokens(b, max_depth=64)
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
    print("\nComputing ARI (hybrid GT) on demo circuit:", demo_circ)
    ari = compute_ari_for_circuit(demo_circ)
    print("  ARI (hybrid GT):", ari)
else:
    print("\nNo circuits available for ARI demo.")