
# query-complexity

## Introduction
#### In this repository, our goal is to implement Semi Definite Programs (SDPs) for the following purposes:
- Finding the exact **query complexity** complexity for boolean functions.
- Designing *optimal* **quantum algorithms** for computing these functions.


## Overview and Sources

### Polynomial Method

#### - Fran's method
File : `tensor1.py`
- https://arxiv.org/abs/2407.13716

#### - Gribling-Laurent SDP

File: `tensor2.py`
- https://arxiv.org/abs/1901.04921


### Adversary Method

#### - Primal and Dual (without symmetry reduction)

File: `adversary.py`

- section 6.2.4 and 6.2.5 : https://ir.cwi.nl/pub/32884/32884D.pdl

#### - Symmetry reduced primal


File: `adversary_symmetry_reduced.py`

- step 1, 1 1/2 and 2 : https://arxiv.org/abs/1007.2905
- block diagonalization: https://quantum-journal.org/papers/q-2024-04-30-1318/pdf/ and https://homepages.cwi.nl/~lex/files/codes.pdf


## Implementation details

### Polynomail Method

### Adversary Method 

#### - Primal 

For a better implementation, we used the following reformulation (as shown in Arjan's thesis, Section 6.2.4)





Implementation of step 1/2:
