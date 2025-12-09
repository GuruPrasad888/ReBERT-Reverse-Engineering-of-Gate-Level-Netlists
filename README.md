# ReBERT – Reverse Engineering BERT for Digital Logic Netlists  
Hybrid Structural + Functional Grouping on ISCAS89 and EPFL Benchmarks

This repository contains two complete pipelines for reverse-engineering digital circuits using a Transformer-based model (**ReBERT**):

1. **ISCAS89 Pipeline**  
2. **EPFL AIG Pipeline**

Each pipeline includes:
- Netlist parsing  
- Structural token extraction  
- Functional simulation  
- Hybrid clustering (functional + structural Jaccard)  
- Bit-pair dataset generation  
- Transformer (ReBERT) training  
- Leave-One-Out (LOO) experiments  
- ARI evaluation for word-level reconstruction  


