
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

#### - Primal: `adversary_primal()`

Takes as input:

    - `n : int` the size of the input. 
    - `f_values : dict` the dictionnary containing the input and output values for the function. The inputs are lists of `0` and `1`s (or `-1` and `1`s), the input are `0` or `1` (could also be `-1` and `1`).
    - `solver` the name of the solver, SCS by default.
    - `verbose` as parameter for the solver

Outputs:


For a better implementation, we used the following reformulation (as shown in Arjan's thesis, Section 6.2.4).


Let $f : \mathcal{D} \to \{0,1\}$, with $\mathcal{D} \subseteq \{0,1\}^n$. Then, $\mathrm{ADV}^{\pm}(f)$ is the optimal value of the following SDP:

$\begin{align}
\max \quad & \sum_{x,y \in \mathcal{D}} \Gamma[x,y], \tag{6.2.1a} \\
\text{s.t.} \quad & \text{diag}(\beta) - \Gamma \circ \Delta_j \succeq 0, \quad \forall j \in [n], \tag{6.2.1b} \\
& \Gamma[x,y] = 0, \quad \forall x,y \in \mathcal{D} \text{ such that } f(x) = f(y), \tag{6.2.1c} \\
& \sum_{x \in f^{-1}(1)} \beta[x] = \tfrac{1}{2}, \tag{6.2.1d} \\
& \sum_{y \in f^{-1}(0)} \beta[y] = \tfrac{1}{2}. \tag{6.2.1e}
\end{align}$

#### - Dual: `adversary_dual()`

Takes as input:

    - `n : int` the size of the input. 
    - `f_values : dict` the dictionnary containing the input and output values for the function. The inputs are lists of `0` and `1`s (or `-1` and `1`s), the input are `0` or `1` (could also be `-1` and `1`).
    - `solver` the name of the solver, SCS by default.
    - `verbose` as parameter for the solver

Outputs:


We use the version shown in Arjan's thesis, Section 6.2.5.


Let $f : \mathcal{D} \to \{0,1\}$, with $\mathcal{D} \subseteq \{0,1\}^n$. Then, $\mathrm{ADV}^{\pm}(f)$ is the optimal value of the following SDP:
$\begin{align}
\min \quad & \max_{x \in \mathcal{D}} \sum_{j=1}^n X_j[x,x], \tag{6.2.4a} \\
\text{s.t.} \quad 
& \sum_{\substack{j=1 \\ x_j \neq y_j}}^n X_j[x,y] = 1, 
\quad \forall x,y \in \mathcal{D},\ f(x) \neq f(y), \tag{6.2.4b} \\
& X_j \succeq 0, \quad \forall j \in [n], \tag{6.2.4c}
\end{align}$
where the optimization ranges over all positive semidefinite matrices 
$X_1, \ldots, X_n \in \mathbb{R}^{\mathcal{D} \times \mathcal{D}}$.







Implementation of step 1/2:
