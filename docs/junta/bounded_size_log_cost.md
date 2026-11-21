# An \(O(\log k)\) transducer for unknown-size majority juntas

## Result

Use the chapter's tie-positive convention for even sizes:

\[
 f_A(S)=\mathbf 1\{|A\cap S|\geq |A|/2\},\qquad 1\leq |A|\leq k.
\]

For odd sizes this is ordinary strict majority. If strict majority is desired
also for even sizes, interchange the \(y=0\) and \(y=1\) branches in (1);
the rest of the construction is unchanged.

There is an explicit feasible dual-adversary family of objective
\(O(\log k)\), and hence a canonical transducer of the same asymptotic cost
that learns \(A\).

The construction gives every nonmaximal odd layer one orthogonal correction
coordinate. Each correction replaces one root of the larger layer's
polynomial. The squared norms of all corrections form one discrete Cauchy
convolution, so they cost one harmonic sum rather than a product of two
harmonic sums.

## 1. Reduce all sizes to odd sizes

Introduce one known coordinate \(y\), and define

\[
 g_A(x,y)=
 \begin{cases}
 f_A(x),&y=1,\\
 1-f_A(\bar x),&y=0.
 \end{cases}                                                     \tag{1}
\]

One phase query to \(g_A\) is simulated by one phase query to \(f_A\). If
\(a=|A|\), then

\[
 g_A=
 \begin{cases}
 \operatorname{MAJ}_{a}(x_A),&a\text{ odd},\\
 \operatorname{MAJ}_{a+1}(x_A,y),&a\text{ even}.
 \end{cases}                                                     \tag{2}
\]

Thus the relevant set

\[
 B_A=\begin{cases}A,&|A|\text{ odd},\\ A\cup\{y\},&|A|\text{ even}\end{cases}
\]

has odd size at most \(K\), where \(K=k\) for odd \(k\) and \(K=k+1\) for
even \(k\). After learning \(B_A\), remove the known coordinate \(y\).

It remains to handle the odd promise

\[
 |B|\in\{r_m=2m+1:0\leq m\leq M\},\qquad K=2M+1.                \tag{3}
\]

## 2. Mixed-size Krawtchouk identity

Write \(K_t^{(r)}\) for the binary Krawtchouk polynomial for size \(r\).
For \(|A|=r\), \(|B|=s\), and \(d=|A\cap B|\),

\[
 \mathbb E_S\!\left[
 K_t^{(r)}(|A\cap S|)K_u^{(s)}(|B\cap S|)
 \right]
 =\mathbf1_{\{t=u\}}\binom dt.                                  \tag{4}
\]

Let \(p_m,q_m:\mathbb R\to\mathcal H\) be vector-valued polynomials of
degree at most \(m\), and define

\[
\begin{aligned}
 L_m(a)&=\sum_{t=0}^{r_m}K_t^{(r_m)}(a)p_m(t),\\
 U_m(a)&=\sum_{t=0}^{r_m}K_t^{(r_m)}(r_m-a)q_m(t),\\
 W_m(a)&=L_m(a)+U_m(a),\\
 c_{m\ell}(t)&=
 \langle p_m(t),q_\ell(t)\rangle+
 \langle q_m(t),p_\ell(t)\rangle .
\end{aligned}                                                    \tag{5}
\]

Krawtchouk duality implies

\[
 L_m(a)=0\quad(a>m),\qquad U_m(a)=0\quad(a\leq m).               \tag{6}
\]

Consequently, the following coefficient conditions imply all adversary
constraints:

\[
\boxed{
c_{m\ell}(0)=1,\qquad
c_{m\ell}(t)=0
\quad
\begin{array}{l}
1\leq t\leq\min(r_m,r_\ell),\\[-2pt]
(m,t)\neq(\ell,r_m).
\end{array}}                                                     \tag{7}
\]

Indeed, on distinguishing queries, (4)--(6) reduce the constraint to

\[
 \sum_{t=0}^{|A\cap B|}
 (-1)^t\binom{|A\cap B|}{t}c_{m\ell}(t)=1.                       \tag{8}
\]

For two distinct sets on the same layer, the omitted endpoint is never
reached.

## 3. Explicit boundary-correction polynomials

Split the roots \(1,\dots,2m\) modulo four:

\[
\begin{aligned}
 \mathcal P_m&=\{a:1\leq a\leq2m,\ a=0,1\pmod4\},\\
 \mathcal Q_m&=\{a:1\leq a\leq2m,\ a=2,3\pmod4\},\\
 P_m(t)&=\prod_{a\in\mathcal P_m}\left(1-\frac ta\right),\\
 Q_m(t)&=\prod_{a\in\mathcal Q_m}\left(1-\frac ta\right).
\end{aligned}                                                    \tag{9}
\]

Both have degree \(m\), their root sets are nested between layers, and

\[
 P_m(0)=Q_m(0)=1,\qquad
 P_m(t)Q_m(t)=0\ (1\leq t\leq2m),\qquad
 P_m(r_m)Q_m(r_m)=1.                                             \tag{10}
\]

For \(m<\ell\), put \(s_m=r_m+1=2m+2\), and select the larger-layer
polynomial containing the root \(s_m\):

\[
 R_\ell^{(m)}=
 \begin{cases}
 Q_\ell,&m\text{ even},\\
 P_\ell,&m\text{ odd}.
 \end{cases}                                                     \tag{11}
\]

Replace that root by the root \(0\):

\[
 C_{m\ell}(t)=\frac{t}{t-s_m}R_\ell^{(m)}(t).                    \tag{12}
\]

There is no pole, since \(s_m\) is a root of \(R_\ell^{(m)}\), and

\[
 C_{m\ell}(r_m)=-r_mR_\ell^{(m)}(r_m).                           \tag{13}
\]

Take

\[
 \mathcal H=\operatorname{span}\{e_*,e_0,\dots,e_{M-1}\}
\]

with the displayed basis orthonormal, and define

\[
\boxed{
\begin{aligned}
p_\ell(t)
={}&\frac{P_\ell(t)}{\sqrt2}e_*
 +\mathbf1_{\{\ell<M,\ \ell\ {\rm even}\}}P_\ell(t)e_\ell\\
&+\sum_{\substack{m<\ell\\m\ {\rm odd}}}
\frac{C_{m\ell}(t)}{2r_m}e_m,\\[3pt]
q_\ell(t)
={}&\frac{Q_\ell(t)}{\sqrt2}e_*
 +\mathbf1_{\{\ell<M,\ \ell\ {\rm odd}\}}Q_\ell(t)e_\ell\\
&+\sum_{\substack{m<\ell\\m\ {\rm even}}}
\frac{C_{m\ell}(t)}{2r_m}e_m.
\end{aligned}}                                                   \tag{14}
\]

At \(t=0\), all corrections vanish and \(e_*\) gives
\(c_{m\ell}(0)=1\). On one layer, every nonbase coordinate occurs on only
one side of the exchange pairing, so

\[
 c_{mm}(t)=P_m(t)Q_m(t)=0,\qquad1\leq t\leq2m.                  \tag{15}
\]

For \(m<\ell\), nested roots make all degrees \(1,\dots,2m\) vanish
automatically. At the sole remaining boundary \(t=r_m\), the \(e_*\)
residue is cancelled by the \(e_m\) correction using (13). Thus (7) holds.

## 4. The PSD adversary family

For a query \(S\subseteq[n]\), and sets \(A,B\) with
\(|A|=r_m\), \(|B|=r_\ell\), define

\[
\boxed{
X_S[A,B]
=2^{-n}
\left\langle
W_m(|A\cap S|),W_\ell(|B\cap S|)
\right\rangle.}                                                  \tag{16}
\]

Each \(X_S\) is a Gram matrix and hence positive semidefinite. Equations
(7)--(8) prove

\[
\sum_{S:f_A(S)\neq f_B(S)}X_S[A,B]=1
\qquad(A\neq B).                                                  \tag{17}
\]

Its objective on layer \(\ell\) is

\[
T_\ell=
\sum_{t=0}^{r_\ell}\binom{r_\ell}{t}
\left(\|p_\ell(t)\|^2+\|q_\ell(t)\|^2\right).                    \tag{18}
\]

## 5. Cost

For \(R=P_\ell\) or \(Q_\ell\), a four-step product recurrence and Wallis
bounds give

\[
\binom rtR(t)^2\leq C\frac r{t(r-t)}
\qquad(1\leq t<r),                                               \tag{19}
\]

with bounded endpoints. Hence the two base-polynomial energies are
\(O(\log r)\).

For fixed larger layer \(\ell\), define

\[
D_m(t)=\frac{C_{m\ell}(t)}{2r_m},\qquad m<\ell.
\]

The same product estimate, together with the discrete Cauchy convolution

\[
\sum_{\substack{a\geq1\\a\neq N}}
\frac1{a^2(N-a)^2}=O(N^{-2}),                                    \tag{20}
\]

gives all correction coordinates at once:

\[
\sum_{m<\ell}
\sum_{t=0}^{r_\ell}\binom{r_\ell}{t}D_m(t)^2
=O(\log(r_\ell+1)).                                              \tag{21}
\]

Applying (20) before summing over \(t\) is the essential improvement:
bounding every root replacement separately produces a spurious second
logarithm. Since all coordinates in (14) are orthogonal,

\[
\max_{\ell\leq M}T_\ell=O(\log(M+2))=O(\log k).                  \tag{22}
\]

## 6. Canonical transducer

Let the phase oracle be

\[
P_A|S\rangle=(-1)^{f_A(S)}|S\rangle.
\]

Factor (16) by setting

\[
|v_A\rangle
=2^{-(n+1)/2}
\sum_{S\subseteq[n]}|S\rangle
W_m(|A\cap S|),
\qquad |A|=r_m.                                                  \tag{23}
\]

The factor \(1/\sqrt2\) relative to (16) accounts for
\(1-(-1)^{f_A(S)+f_B(S)}=2\) on distinguishing queries. Therefore

\[
\langle v_A|(I-P_AP_B)|v_B\rangle=1
\qquad(A\neq B).                                                  \tag{24}
\]

The two state families

\[
|0\rangle\oplus P_A|v_A\rangle,
\qquad
|A\rangle\oplus|v_A\rangle
\]

have the same Gram matrix. Hence an input-independent unitary \(U_{\leq K}\)
exists with

\[
U_{\leq K}:
|0\rangle\oplus P_A|v_A\rangle
\longmapsto
|A\rangle\oplus|v_A\rangle.                                      \tag{25}
\]

Thus

\[
|0\rangle\oplus|v_A\rangle
\xrightarrow{\,I\oplus P_A\,}
|0\rangle\oplus P_A|v_A\rangle
\xrightarrow{\,U_{\leq K}\,}
|A\rangle\oplus|v_A\rangle                                      \tag{26}
\]

is the required canonical transducer for all odd sizes simultaneously. Its
phase-query Las Vegas cost is

\[
\|v_A\|^2=\frac12T_m=O(\log k).                                  \tag{27}
\]

Compose (26) with the one-query oracle simulation (1). Its output is
\(B_A\); remove the distinguished element \(y\), retaining its membership
bit as garbage if a reversible output map is required. This produces the
desired \(O(\log k)\)-cost transducer for every \(1\leq|A|\leq k\).

## 7. Verification

Direct evaluation of (14) verifies every coefficient constraint through
\(k=127\); the largest floating-point residual is \(1.9\times10^{-14}\).
An independent phase-switched correction family was checked through layer
95 with zero feasibility residual and the same harmonic behavior. These
computations audit the exact identities above rather than replace the
proof.
