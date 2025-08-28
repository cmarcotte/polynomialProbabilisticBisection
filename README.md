# Polynomial Probabilistic Bisection Algorithm (PPBA)

This is an implementation of the PBA following Waeber _et al_ (2013) [^1].

The method uses the prior belief $f: x \in \Omega \to \mathbb{R}^+$ for $\Omega = (a, b) \subset \mathbb{R}$ for the location of the sign-change $x^*$ (hereafter, I'll refer to $x^*$ as a root, but the method works for the less specific condition).

The procedure requires an oracle, $Z: \Omega \to \mathbb{B}$, where $\mathbb{B} = \{0,1\}$ is a boolean, which returns $Z(x_i) = z_i = 1$ if input $x$ has overshot the sign-change $x^*$, with _constant_ probability $1/2 < p \leq 1$, otherwise $z_i = 0$ with probability $p$.

We define the CDF of the belief $F(x) = (b-a)^{-1}\int_{a}^{x} f(x^\prime) \mathrm{d}x^\prime$, and define

$$
	\gamma(x) = (1 - p) F(x) + p (1 - F(x)).
$$

The update to the belief with each new sample $x_i$ and oracle value $z_i = Z(x_i)$ is given by:

$$
	f(y | x_i) =
		\begin{cases}
			\begin{cases}
				p f(y)/\gamma(x_i), &y \geq x_i \\
				(1 - p) f(y) / \gamma(x_i), &y < x_i \\
			\end{cases} &z_i = 1, \\
			\begin{cases}
				(1 - p) f(y) / (1 - \gamma(x_i), &y \geq x_i\\
				p f(y) / (1 - \gamma(x_i)), &y < x_i
			\end{cases} &z_i = 0,
		\end{cases}
$$

### Example

Let $f(x) = \mathcal{U}(0,1) = 1$, with fixed $1/2 < p \leq 1$, and let $x_1 \in (0,1)$. Then $F(x_1) = x_1$, and $\gamma(x_1) = (1 - p) x_1 + p (1 - x_1)$.
Let us assume that $z_1 = 1$; then the updated belief is:

$$
	f(x) =
		\begin{cases}
			\frac{(1-p)}{(1-p)x_1 + p(1-x_1)}, &x < x_1, \\
			\frac{p}{(1-p)x_1 + p(1-x_1)}, &x \geq x_1
		\end{cases},
$$
i.e., it shifts the previously uniformly distributed probability density to the left and right of $x_1$ according to the oracle.

## Implementation

One should specify a prior belief $f(x)$ -- this is typically just an uninformative uniform distribution over $\Omega: f(x) = \mathcal{U}(a,b)$.
In principle, one can approximate any prior with a piecewise polynomial (spline) $S \in \mathbb{C}^{p_i}(a_i, b_i)$ input, if desired.
The module uses an in-built `SparsePolyDistribution{T}` type to represent the belief $f$, which builds off [ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl) to form a particular subset of `Fun`s; specifically, we require inputs `f` which satisfy:
```julia
@test domain(f) isa 𝒟
@test space(f) isa 𝒮
@test f isa ℱ
```
where 
```julia
const 𝒟 = UnionDomain{T,NTuple{N,Segment{T}}} where {N,T<:Real}
const 𝒮 = PiecewiseSpace{NTuple{N,Chebyshev{Segment{T},T}},𝒟{N,T},T} where {N,T<:Real}
const ℱ = Fun{𝒮{N,T},T,Vector{T}} where {N,T<:Real}
```

That is, for $\Omega = \cup_{i=1}^{N}\Omega_i$ and $\Omega_i = (a_i, b_i)$, we require inputs which are expressed as Chebyshev polynomial expansions $f =  \cup_{i=1}^{N} f_i$ $f_i: \Omega_i \to \mathbb{R}^+$. Generally, it is advised that the order of the expansion on each segment be limited -- the canonical case is _constant_ -- but expansions up to 32 modes appears to work well enough.

To update the prior, we require an evaluation of the CDF of the belief $F(x) = (b-a)^{-1}\int_{a}^{x} \mathrm{d}x' f(x')$ at the sample point.
The CDF $F(x)$ is represented as a `SparsePolyCumulativeDistribution{T}` which is not required to be normalized (like the `SparsePolyDistribution{T}`) and is $\mathcal{C}^{p+1}$ on $\Omega_i$ for $f_i \in \mathcal{C}^{p}$, i.e. $F =  \cup_{i=1}^{N} F_i$ where $F_i: \Omega_i \to \mathbb{R}^+$.

The bisection procedure will then sample the initial domain
according to the median rule (i.e. $x_1 = x: F(x) = \frac{1}{2}$), and update the belief according to the oracle values returned from these sample points $z_i = Z(x_i)$: $f(x | \{z_i\}_{i=1}^{n})$.

The method decomposes the `domain(f)` on each update, bisects the `Segment` which contains the sample point, appends the newly bisected `Segment`s, drops the original `Segment`, and then re-samples `f` on the union of the `Segment`s.
Further, because the `SparsePoly

[^1]: [Waeber, Rolf, Peter I. Frazier, and Shane G. Henderson. "Bisection search with noisy responses." SIAM Journal on Control and Optimization 51, no. 3 (2013): 2261-2279.](https://doi.org/10.1137/120861898)

## Comments

Since this type allows arbitrarily high-order polynomial expansions $p$ [^2], and the update to the belief involves only multiplicative factors on new subdomains, we will find that the degree of $f$ on each `Segment{T}` is (approximately!) _constant_ with each iteration and the number of `Segment{T}`s increases by one on each iteration.
This is reasonably performant in practice, despite the substantial increase in complexity compared to [probabilisticBisection](https://github.com/cmarcotte/probabilisticBisection).
However, the memory usage compared to the $f \in \mathcal{C}^0$ method is substantial due to the creation of new `Segment{T}` and `Fun` objects throughout the update. Making this appropriately optimized using mutable state is ongoing work.

[^2]: That is, `ApproxFun.Fun` tops out at `length(coefficients(f)) == 1 + 2^20`, which might as well be arbitrarily high.
