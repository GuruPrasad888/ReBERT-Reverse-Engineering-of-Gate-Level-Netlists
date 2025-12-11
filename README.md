# Structural Similarity Learning for Hardware Reverse Engineering

**BERT-based Word-Level Grouping on ISCAS89 and EPFL Benchmarks**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements a deep learning approach for reverse-engineering digital circuits using structural similarity learning with BERT-based Transformers. The system learns to group individual bit-level signals into word-level structures by analyzing their structural patterns.

## 📋 Overview

This work presents a **purely structural approach** to hardware reverse engineering that:
- Extracts bounded-depth logic cones (K=6) from gate-level netlists
- Encodes circuit structure as token sequences via preorder traversal
- Trains a BERT-based classifier using **self-supervised structural similarity labels**
- Achieves 91.23% accuracy on ISCAS89 sequential circuits
- Achieves 99.91% accuracy on EPFL arithmetic benchmarks

## 🏗️ Architecture

The system consists of two complete pipelines:

### 1. **ISCAS89 Pipeline** (`Rebert(ISCAS).py`)
For sequential circuits with flip-flops:
- Parses gate-level Verilog (AND, OR, NAND, NOR, NOT, BUF)
- Extracts structural cones from FF D-inputs
- 20-token vocabulary with sequential primitives
- Max sequence length: 64 tokens

### 2. **EPFL AIG Pipeline** (`Rebert(EPFL).py`)
For arithmetic circuits in AIG form:
- Parses AIG-style Verilog (AND, OR, NOT only)
- Builds bounded-depth expression trees for primary outputs
- 7-token vocabulary (simplified for AIG)
- Max sequence length: 522 tokens

## 🔧 Pipeline Stages
```
Verilog Netlist
      ↓
┌─────────────────────────┐
│  Netlist Parsing        │
│  - Extract gates/FFs    │
│  - Build driver map     │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  Bounded Cone           │
│  Extraction (K=6)       │
│  - Preorder traversal   │
│  - Depth-limited        │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  Token Sequence         │
│  Generation             │
│  - [CLS] + tokens + [SEP]│
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  Bigram Jaccard         │
│  Similarity             │
│  - Pos: J ≥ 0.6        │
│  - Neg: J ≤ 0.3        │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  BERT Training          │
│  - 4 layers, 4 heads    │
│  - Hidden size: 256     │
│  - Binary classification│
└─────────────────────────┘
      ↓
  Learned Similarity
```

## 📊 Results

### ISCAS89 Sequential Circuits (24 benchmarks)
- **Total training pairs:** 48,914 (46.5% pos, 53.5% neg)
- **Final validation accuracy:** 91.23%
- **Training epochs:** 5
- **Largest circuit:** s38584 (19,253 gates, 1,426 FFs)

### EPFL Arithmetic Circuits (7 benchmarks)
- **Total training pairs:** 21,814 (98.0% pos, 2.0% neg)
- **Final validation accuracy:** 99.91%
- **Training epochs:** 5
- **Largest circuit:** div (40,648 assignments, 128 outputs)

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch transformers scikit-learn numpy
```

### ISCAS89 Training
```python
# Update NETLIST_DIR in rebert_iscas89.py
NETLIST_DIR = "/path/to/iscas89_netlists"

# Run training
python rebert_iscas89.py
```

### EPFL Training
```python
# Update NETLIST_DIR in rebert_epfl.py
NETLIST_DIR = "/path/to/epfl_netlists"

# Run training
python rebert_epfl.py
```


## 🔬 Technical Details

### Self-Supervised Pair Labeling

Unlike supervised learning approaches, this implementation generates training labels automatically using structural similarity:
```python
# Bigram-based Jaccard similarity
bigrams_i = {(tok[k], tok[k+1]) for k in range(len(tok)-1)}
bigrams_j = {(tok[k], tok[k+1]) for k in range(len(tok)-1)}

J = len(bigrams_i ∩ bigrams_j) / len(bigrams_i ∪ bigrams_j)

# Labeling
if J ≥ 0.6:    label = POSITIVE (same word)
if J ≤ 0.3:    label = NEGATIVE (different words)
else:          DISCARD (ambiguous)
```

### Model Architecture

**ISCAS89 Model:**
- Vocabulary: 20 tokens
- Embedding dim: 256
- Transformer layers: 4
- Attention heads: 4
- FFN hidden: 1024
- Total parameters: ~1.5M

**EPFL Model:**
- Vocabulary: 7 tokens
- Embedding dim: 256
- Max sequence: 522
- Architecture: Same as ISCAS89

### Bounded-Depth Cone Extraction

To prevent exponential blowup in large circuits, cone extraction is limited to depth K=6:
```python
def build_cone_tokens(root_net, max_depth=6):
    if depth >= max_depth:
        return ['DEPTH_CUT']
    # ... recursive logic expansion
```

## 📈 Detailed Results

### ISCAS89 Training Progress
| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|----------|--------------|
| 1     | 0.3051    | 0.2595   | 90.72%       |
| 2     | 0.2481    | 0.2616   | 91.25%       |
| 3     | 0.2424    | 0.2497   | 91.56%       |
| 4     | 0.2405    | 0.2462   | 91.15%       |
| 5     | 0.2408    | 0.2480   | **91.23%**   |

### EPFL Training Progress
| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|----------|--------------|
| 1     | 0.0339    | 0.0117   | 99.66%       |
| 2     | 0.0155    | 0.0109   | 99.91%       |
| 3     | 0.0152    | 0.0110   | 99.91%       |
| 4     | 0.0152    | 0.0107   | 99.91%       |
| 5     | 0.0151    | 0.0107   | **99.91%**   |

## ⚠️ Limitations

1. **Extreme class imbalance in EPFL:** 98% positive pairs limit negative sample diversity
2. **No functional information:** Purely structural approach may miss functional equivalences
3. **Fixed thresholds:** Jaccard thresholds (0.6, 0.3) not adaptive to circuit characteristics
4. **No clustering evaluation:** Current results show only binary classification accuracy
5. **Scale disparity:** Large circuits dominate training (92.5% of ISCAS89 data)

## 🔮 Future Directions

- [ ] Implement tree-based positional embeddings (as in original ReBERT)
- [ ] Add functional simulation for hybrid structural-functional approach
- [ ] Adaptive threshold optimization based on circuit characteristics
- [ ] Leave-one-out cross-validation for generalization testing
- [ ] Clustering evaluation using ARI/NMI metrics
- [ ] Multi-scale cone extraction (varying K)
- [ ] Data augmentation via gate-level corruption

## 📚 Citation

If you use this code, please cite:

**Original ReBERT Paper:**
```bibtex
@inproceedings{zhang2025rebert,
  title={ReBERT: LLM for Gate-Level to Word-Level Reverse Engineering},
  author={Zhang, Lizi and Davoodi, Azadeh and Topaloglu, Rasit Onur},
  booktitle={2025 Design, Automation \& Test in Europe Conference (DATE)},
  pages={1--7},
  year={2025},
  organization={IEEE}
}
```

**This Implementation:**
```bibtex
@misc{venkatesan2025structural,
  title={Structural Similarity Learning for Hardware Reverse Engineering},
  author={Venkatesan, Guruprasad},
  year={2025},
  institution={University of Texas at Dallas}
}
```

## 📧 Contact

**Guruprasad Venkatesan**  
Department of Electrical and Computer Engineering  
The University of Texas at Dallas  
Email: GuruPrasad.Venkatesan@utdallas.edu

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original ReBERT architecture by Zhang et al. (DATE 2025)
- ISCAS89 benchmark suite
- EPFL arithmetic benchmark suite
- Hugging Face Transformers library
- PyTorch team

---

**Note:** This implementation represents a **structural similarity learning** approach and differs from the original supervised ReBERT methodology. See the comparison table above for detailed differences.