# Numerically optimizing multiple biases in the level-0 junta SDP

This README explains the code in `level0_multibias.py` and the convex program it solves.

The main use case is **EXACT\(_\ell\)** junta learning,
\[
f_A(S)=\mathbf 1\{|A\cap S|=\ell\},\qquad |A|=k,
\]
including the two important endpoints:

- **EXACT-HALF**: the optimum has the familiar one-bias `p = 1/2` parity certificate;
- **OR**: up to complementing the oracle output, OR is EXACT\(_0\), and Belovs' construction uses a continuum of biases. The code can either optimize the unrestricted level-0 multi-bias problem or reproduce Belovs' continuum exactly by finite quadrature.

The implementation uses the **direct distance / local-weight basis**, so no numerical Krawtchouk or Hahn normalization is needed.

---

## 1. Fixed bias `p`

Take two hidden `k`-sets `A,B` at Johnson distance
\[
d=|A\setminus B|=|B\setminus A|\in\{1,\ldots,k\}.
\]
For a Bernoulli-`p` query `S`, put
\[
a=|A\cap S|,\qquad b=|B\cap S|.
\]
Write
\[
J\sim\operatorname{Bin}(k-d,p),\qquad
Y,Z\sim\operatorname{Bin}(d,p)
\]
independently. Then
\[
a=J+Y,\qquad b=J+Z.
\]
Hence
\[
\mu_{d,p}(a,b)
=
\Pr(a,b)
=
\sum_j
\binom{k-d}{j}
\binom d{a-j}
\binom d{b-j}
 p^{a+b-j}(1-p)^{k+d-a-b+j}.
\]
The one-input marginal is
\[
w_p(a)=\binom ka p^a(1-p)^{k-a}.
\]

For EXACT\(_\ell\), one output class is the singleton `{ell}`. The reduced cross block is therefore one column. Normalize it by defining
\[
z_p(a)=\sqrt{w_p(a)w_p(\ell)}\,C_p(a,\ell),\qquad a\ne\ell.
\]
Define the `k x k` matrix
\[
\boxed{
R_{k,\ell}(p)_{d,a}
=
\frac{\mu_{d,p}(a,\ell)}
{\sqrt{w_p(a)w_p(\ell)}}
}
\]
with rows `d=1,...,k` and columns `a != ell`.

A single bias contributes the distance profile
\[
2R(p)z_p\in\mathbb R^k.
\]
There is **no requirement** that this vector equal `1` by itself.

---

## 2. Several biases: the finite level-0 SOCP

Choose biases
\[
P=\{p_1,\ldots,p_m\}.
\]
Each bias gets its own level-0 vector `z_j`. Exact feasibility requires only
\[
\boxed{
2\sum_{j=1}^m R(p_j)z_j=\mathbf 1_k.
}
\]
The cost is the sum of the rank-one cross-block costs,
\[
\boxed{
U_{0,P}(k,\ell)
=
\min_{z_1,\ldots,z_m}
2\sum_{j=1}^m\|z_j\|_2
\quad\text{s.t.}\quad
2\sum_jR(p_j)z_j=\mathbf1_k.
}
\]
This is a second-order cone program (SOCP), i.e. the singleton-output specialization of the level-0 nuclear-norm SDP.

After solving it, recover the actual local kernel coefficients from
\[
\boxed{
C_j(a,\ell)
=
\frac{z_j(a)}{\sqrt{w_{p_j}(a)w_{p_j}(\ell)}}.
}
\]
Thus `C_j` depends on the local weight `a` (and the singleton `b=ell`) and is allowed to have different numerical coefficients for different bias labels `p_j`.

The code contains an optional direct CVXPY solver:

```python
solve_exact_primal_cvxpy(k, ell, biases)
```

CVXPY is not needed for the tested exchange / quadrature routines.

---

## 3. Continuum of biases

The mathematically clean continuum object is a **vector-valued measure** `Z(dp)` on `(0,1)` rather than an ordinary function. The continuous atomic program is
\[
\boxed{
U_{0,\infty}(k,\ell)
=
2\inf_Z |Z|((0,1))
}
\]
subject to
\[
\boxed{
2\int_0^1R(p)\,dZ(p)=\mathbf1_k.
}
\]
If the measure is atomic,
\[
dZ(p)=\sum_j z_j\,\delta_{p_j}(dp),
\]
this reduces exactly to the finite program above.

So "allow a continuum of biases" means that the optimizer may choose an arbitrary measure on `p`. Because the target lives in finite dimension `R^k`, an optimal or arbitrarily close solution can usually be represented by finitely many atoms. Therefore an **unrestricted optimizer should not be expected to literally return a continuous density**, even when Belovs' analytic OR certificate is written as an integral.

---

## 4. The continuum dual

The dual of the atomic program is especially simple:
\[
\boxed{
U_{0,\infty}(k,\ell)
=
\max_{y\in\mathbb R^k}\mathbf1^Ty
\quad\text{s.t.}\quad
\|R(p)^Ty\|_2\le1
\quad\forall p\in(0,1).
}
\]
Define the dual envelope
\[
F_y(p)=\|R(p)^Ty\|_2.
\]
Then continuum feasibility is just
\[
F_y(p)\le1\qquad\text{for every }p.
\]
The biases used by an optimal primal atomic solution occur at **contact points** where
\[
F_{y_*}(p)=1.
\]
This is why the dual is a convenient way to discover useful biases.

---

## 5. Exchange / cutting-plane optimization over the continuum

The command

```bash
python -m junta_learning.multibias exchange --k 12 --ell 0
```

implements the following algorithm.

1. Start with a small bias set, by default `P={0.5}`.
2. Solve the finite dual
   \[
   \max \mathbf1^Ty
   \quad\text{s.t.}\quad
   \|R(p_j)^Ty\|_2\le1,\quad p_j\in P.
   \]
3. Search the full interval for
   \[
   p_*\in\arg\max_pF_y(p).
   \]
4. If `F_y(p_*) > 1 + tolerance`, add `p_*` to `P` and repeat.
5. Stop when the whole scanned continuum is approximately dual-feasible.

The code uses a dense one-dimensional scan plus local scalar refinement. Only the finite dual is optimized with SciPy/SLSQP.

### A useful numerical bracket

Let
\[
v=\max_pF_y(p).
\]
The finite restricted problem has value approximately
\[
U_{0,P}=\mathbf1^Ty.
\]
Since allowing more biases can only help,
\[
U_{0,\infty}\le U_{0,P}.
\]
Also `y/v` is continuum-dual-feasible, so
\[
\boxed{
\frac{\mathbf1^Ty}{v}
\le U_{0,\infty}
\le \mathbf1^Ty.
}
\]
This is what the script reports as the scaled continuum lower bound and restricted finite-bias upper bound.

**Important:** the biases accumulated by the exchange routine are constraints used to certify the continuum dual. They are not automatically all active in the final primal. To obtain the actual primal `z_j` vectors, solve the finite primal on the final/contact bias set, e.g. with `solve_exact_primal_cvxpy`.

---

## 6. Why the middle collapses to `p=1/2`

For even `k`, set `ell=k/2`. At `p=1/2`, define
\[
z(a)=
\begin{cases}
\sqrt{w(a)/w(\ell)},&a\not\equiv\ell\pmod2,\\
0,&a\equiv\ell\pmod2.
\end{cases}
\]
Then exactly
\[
2R(1/2)z=\mathbf1_k.
\]
Its cost is
\[
2\|z\|_2
=
\sqrt{\frac{2}{w_{1/2}(\ell)}}
=
\sqrt{\frac{2^{k+1}}{\binom{k}{k/2}}}.
\]
So a **single bias `p=1/2` is already a complete feasible level-0 certificate**.

Run

```bash
python -m junta_learning.multibias exact-half --k 12
```

Tested output:

```text
EXACT-half k=12, p=1/2
cost                    : 2.97754947328
closed-form cost        : 2.97754947328
max feasibility residual: 3.997e-15
```

This is the numerical manifestation of the central parity solution.

---

## 7. OR and Belovs' continuum of biases

OR is equivalent, up to complementing the output bit, to EXACT\(_0\). The unrestricted command

```bash
python -m junta_learning.multibias exchange --k 12 --ell 0
```

therefore optimizes the full level-0 multi-bias problem for OR/EXACT\(_0\).

Near this endpoint the exchange method keeps discovering useful `p` values spread across the interval rather than collapsing to `1/2`. With moderate tolerances this already gives a value of order `sqrt(k)`, as expected.

However, Belovs' original OR proof uses a particular analytic continuum, not necessarily the unrestricted level-0 optimum. Its fixed-`p` distance profile is
\[
\boxed{
v_d(p)=d(1-p)^{d-1}.
}
\]
Uniform integration gives exact feasibility:
\[
\boxed{
\int_0^1 d(1-p)^{d-1}\,dp=1
\qquad(d=1,\ldots,k).
}
\]
Thus the continuum measure is simply `dp`.

### Exact finite compression of the continuum

Since `v_d(p)` is a polynomial of degree `d-1 <= k-1`, an `m`-point Gauss-Legendre rule with
\[
2m-1\ge k-1,
\qquad
m=\lceil k/2\rceil,
\]
reproduces **all feasibility equations exactly**:
\[
\sum_{j=1}^m\alpha_jd(1-p_j)^{d-1}=1,
\qquad d=1,\ldots,k.
\]
So Belovs' continuum can be replaced, for fixed `k`, by only `ceil(k/2)` biases with positive weights.

Run

```bash
python -m junta_learning.multibias belovs-or --k 12
```

Tested output:

```text
Belovs OR, k=12, m=6
max feasibility residual : 9.992e-16
finite quadrature cost bd: 9.95565628461
continuum pi*sqrt(k) bd  : 10.8827961854
bias, quadrature weight
  0.033765243  0.085662246
  0.169395307  0.180380787
  0.380690407  0.233956967
  0.619309593  0.233956967
  0.830604693  0.180380787
  0.966234757  0.085662246
```

As `k` grows, `m=ceil(k/2)` grows and the quadrature nodes fill `[0,1]`. This is a precise finite-dimensional way to "reproduce the continuum" numerically.

Notice the conceptual distinction:

- **Belovs OR certificate:** a prescribed uniform continuum `dp`; Gauss quadrature compresses it exactly for the finite set of distance equations.
- **Optimal unrestricted level 0:** optimizes over all vector-valued bias measures and may choose a much sparser atomic measure.

The latter can improve the constant because it is not required to keep Belovs' fixed per-bias channel shape.

---

## 8. General symmetric profile: nuclear-norm SDP

For a general Boolean profile
\[
h:\{0,\ldots,k\}\to\{0,1\},
\]
put
\[
W_0=h^{-1}(0),\qquad W_1=h^{-1}(1).
\]
For each distance `d` and bias `p`, define the normalized matrix
\[
R_{d,p}(a,b)
=
\frac{\mu_{d,p}(a,b)}{\sqrt{w_p(a)w_p(b)}},
\qquad a\in W_0,\ b\in W_1.
\]
For finitely many biases, the full level-0 program is
\[
\boxed{
\begin{aligned}
\min\quad&2\sum_j\|B_j\|_*\\
\text{s.t.}\quad&
2\sum_j\langle R_{d,p_j},B_j\rangle=1,
\qquad d=1,\ldots,k.
\end{aligned}
}
\]
The code implements this optional CVXPY routine as

```python
solve_general_level0_cvxpy(k, profile, biases)
```

For EXACT, `B_j` has one column and `||B_j||_* = ||z_j||_2`, which is why the specialized code is an SOCP.

---

## 9. Relation to the electrical construction

The attached EXACT\(_\ell\) proof uses the more structured ansatz
\[
z(p)=A_p\phi_\ell,
\]
where
\[
\phi_\ell
=
\left(\int A_p^TA_p\,dp\right)^+b.
\]
This automatically solves the multiple-bias equality
\[
\int A_p^Tz(p)\,dp=b.
\]
It is a beautiful analytic, rank-one, level-0 certificate, but it restricts all bias channels to be generated by one common potential `phi`. The unrestricted atomic SOCP above allows the direction of `z(p)` to vary independently with `p`, so it can only improve the numerical level-0 optimum.

The two approaches are therefore:

1. **electrical / common-potential ansatz:** analytically tractable and already gives the desired `O(sqrt(k)/t^(1/4))` bound;
2. **unrestricted multi-bias level 0:** best numerical level-0 upper bound, found by the SOCP / semi-infinite dual.

---

## 10. Suggested numerical workflow

For a given `k,ell`:

```bash
# 1. If ell=k/2, verify that one p=1/2 channel works.
python -m junta_learning.multibias exact-half --k 12

# 2. Approximate the unrestricted continuum optimum.
python -m junta_learning.multibias exchange --k 12 --ell 3 --scan 601 --tol 1e-4 --rounds 30

# 3. OR endpoint: compare unrestricted level 0 with Belovs' analytic continuum.
python -m junta_learning.multibias exchange --k 12 --ell 0 --scan 601 --tol 1e-4 --rounds 30
python -m junta_learning.multibias belovs-or --k 12
```

For publication-quality numerics, use the exchange routine to discover the contact biases and then solve the corresponding finite primal with a conic solver (CVXPY + CLARABEL/MOSEK/SCS, for example) to recover the actual `z_j` or `B_j` and certify primal/dual agreement.

---

## 11. The picture to look for

As `ell` moves from the center toward the edge:

- at `ell=k/2`, a single atom at
  \[
  p=1/2
  \]
  gives the parity solution;
- off center, more contact biases become useful;
- at the OR edge, Belovs gives the particularly simple continuous representation
  \[
  d\nu(p)=dp,
  \]
  while the unrestricted finite-dimensional level-0 optimizer can compress the necessary information into finitely many atoms.

Thus the continuum of biases is not a new hierarchy level. It is an enrichment of the **query-distribution channel inside level 0**. The representation-theoretic level remains zero throughout.
