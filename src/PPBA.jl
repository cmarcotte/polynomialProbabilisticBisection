module PPBA

export SparsePolyDistribution, SparsePolyCumulativeDistribution, bisection, sum, cumsum

import ApproxFunOrthogonalPolynomials: Chebyshev
import ApproxFunBase: Interval, Segment, PiecewiseSpace, roots, domain, space, sum, cumsum, Fun, SumSpace, components, cfstype
import DomainSets: UnionDomain
#=
# These consts are \scrD[tab], \scrS[tab], \scrF[tab], and describe the allowable subsets of
# domains, spaces, and funs for the PBA with ApproxFunExt.
# These are different from, for example:
# 		D = UnionDomain{T, Vector{Segment{T}}} where {T<:Real}
# 		S = PiecewiseSpace{Vector{Chebyshev{Segment{T}, T}}, D{T}, T} where {T<:Real}
# 		F = Fun{S{T},T,Vector{T}} where {T<:Real}
# in that, if d = UnionDomain([Segment(a,r), Segment(r,b)]) then d isa D, and
# if s = PiecewiseSpace(Chebyshev{Segment{T},T}[Chebyshev(dd) for dd in d.domains]) then !(s isa S), instead s isa 𝒮.
#
# That is, you can't actually create a Fun over a vector of segments, only a tuple of segments.
=#
const 𝒟 = UnionDomain{T,NTuple{N,Segment{T}}} where {N,T<:Real}
const 𝒮 = PiecewiseSpace{NTuple{N,Chebyshev{Segment{T},T}},𝒟{N,T},T} where {N,T<:Real}
const ℱ = Fun{𝒮{N,T},T,Vector{T}} where {N,T<:Real}

function normalize!(f::ℱ)
   f .= f ./ sum(f)
   return nothing
end

abstract type AbstractSparseDistribution end

struct SparsePolyDistribution{T} <: AbstractSparseDistribution
   f::ℱ{N,T} where {N,T<:Real}
   function SparsePolyDistribution(f::ℱ{N,T}) where {N,T<:Real}
      if !(sum(f) ≈ one(T))
         normalize!(f)
      end
      return new{T}(f)
   end
end

SparsePolyDistribution(x::NTuple{N,T}) where {N,T<:Real} = SparsePolyDistribution(Fun(x -> one(T), union((Segment(a, b) for (a, b) in zip(x[begin:end-1], x[begin+1:end]))...)))

function SparsePolyDistribution(x::NTuple{2,T}) where {T<:Real}
   a, b = x
   r = (a + b) * T(1 // 2)
   return SparsePolyDistribution((a, r, b))
end

function SparsePolyDistribution(x::Vector{T}) where {T<:Real}
   N = length(x)
   if N >= 3
      return SparsePolyDistribution((x...,))
   elseif N == 2
      a, b = first(x), last(x)
      r = (a + b) * T(1 // 2)
      return SparsePolyDistribution((a, r, b))
   else
      @error "SparsePolyDistribution(x::Vector{T}) must have length(x) >= 2!"
   end
end

function (f::SparsePolyDistribution{T})(x::T) where {T<:Real}
   return f.f(x)
end

function normalize!(f::SparsePolyDistribution{T}) where {T<:Real}
   normalize!(f.f)
   return nothing
end

function sum(f::SparsePolyDistribution{T}) where {T<:Real}
   return sum(f.f)
end

struct SparsePolyCumulativeDistribution{T} <: AbstractSparseDistribution
   f::ℱ{N,T} where {N,T<:Real}
   function SparsePolyCumulativeDistribution{T}(f::ℱ{N,T}) where {N,T<:Real}
      if !(last(f) - first(f) ≈ one(T))
         normalize!(f)
      end
      return new{T}(f)
   end
end

function (F::SparsePolyCumulativeDistribution{T})(x::T) where {T<:Real}
   return F.f(x)
end

function cumsum(f::ℱ{N,T}) where {N,T<:Real}
   #=
   # This function is a bugfix for ApproxFunBase.cumsum(f::Fun{<:PiecewiseSpace})
   =#
   vf = cumsum.(components(f))
   r = zero(T)# to fix for f::Fun{<:PiecewiseSpace}, this line should be r = zero(cfstype(first(vf)))
   for k in eachindex(vf)
      vf[k] += r
      r = last(vf[k])
   end
   Fun(vf, PiecewiseSpace)
end

function cumsum(f::SparsePolyDistribution{T})::SparsePolyCumulativeDistribution{T} where {T<:Real}
   if !(sum(f.f) ≈ one(T))
      normalize!(f.f)
   end
   N = length(domain(f.f).domains)
   _F::ℱ{N,T} = cumsum(f.f)# this is not type-stable if you don't specify the output type
   return SparsePolyCumulativeDistribution{T}(_F)
end

function median(f::ℱ{N,T}) where {N,T<:Real}
   ##return first(roots(f - T(1 // 2))) # StackOverflowError when on a UnionDomain
   for d in domain(f).domains
      Fa = f(d.a)
      Fb = f(d.b)
      if Fa == T(1 // 2)
         return d.a
      elseif Fb == T(1 // 2)
         return d.b
      elseif Fa < T(1 // 2) && Fb > T(1 // 2)
         g = Fun(x -> f(x), Interval(d))# could perhaps be avoided by providing a view of the relevant set of coefficients in F.f
         return T(first(roots(g - T(1 // 2))))# this solves the type instability... by coercion
      end
   end
end

function median(F::SparsePolyCumulativeDistribution{T}) where {T<:Real}
   ##return first(roots(F.f - T(1 // 2))) # StackOverflowError when on a UnionDomain
   return median(F.f)
end

function median(f::SparsePolyDistribution{T}) where {T<:Real}
   F = cumsum(f)
   return median(F)
end

function splitDomainAt(x::T, domain::Segment{T})::NTuple{2,Segment{T}} where {T<:Real}
   left = Segment(domain.a, x)
   right = Segment(x, domain.b)
   return (left, right)
end

function splitDomainAt(x::T, f::SparsePolyDistribution{T})::SparsePolyDistribution{T} where {T<:Real}
   domains = [domain(f.f).domains...]
   for (n, d) in enumerate(domains)
      if x == d.a || x == d.b
         return f
      elseif d.a < x < d.b
         #				n
         # (a, b), (c, d), (e, f)
         # (a, b), (c, d), (c, g), (g, d), (e, f)
         # (a, b), (c, g), (g, d), (e, f)
         left, right = splitDomainAt(x, d)
         if !(isempty(left) && isempty(right))
            insert!(domains, n + 1, left)
            insert!(domains, n + 2, right)
            deleteat!(domains, n)
         end
         return project(f, NTuple{length(domains),eltype(domains)}(domains))
      end
   end
end

function project(f::SparsePolyDistribution{T}, domains::NTuple{N,Segment{T}}) where {T<:Real,N}
   # it is tempting to try and use setdomain(::Fun, ::Domain), but doing so with the UnionDomain does not correctly set the coefficients if a domain is subdivided.
   return SparsePolyDistribution(Fun(x -> f.f(x), union(domains...)))
end

function bisection(Z::Function, interval::NTuple{2,T}, p::T; reltol::T=T(1e-6), abstol::T=zero(T), maxiters::Integer=1024) where {T<:Real}
   # we want to just set up a SparsePolyDistribution{T} by subdividing the initial interval to make the union of two Segments
   a = first(interval)
   b = last(interval)
   r = (b + a) * T(1 // 2)
   _f = Fun(x -> one(T), union(Segment(a, r), Segment(r, b)))
   f = SparsePolyDistribution{T}(_f)
   return bisection!(Z, f, p; reltol=reltol, abstol=abstol, maxiters=maxiters)
end

function bisection!(Z::Function, f::SparsePolyDistribution{T}, p::T; reltol::T=T(1e-6), abstol::T=zero(T), maxiters::Integer=1024) where {T<:Real}
   #=
      let f0(x) ~ U(interval)
      let Z(x::Real)::Bool tell us with probability p if x >= the root (true) or not (false)
      let x1 = median(f0(x)) from the interval be sampled using the oracle, Z1 = Z(x1)
      f1 is then updated using Bayes' rule from f0 (see update!), and
      then f0 <- f1,
      and then you check if the root is converged
      =#
   iters = 0
   x = median(f)
   while !converged(f, x; abstol=abstol, reltol=reltol) && iters < maxiters
      z = Z(x)
      f = update!(f, p, x, z)
      iters += 1
   end
   return median(f), f
end

function converged(f::SparsePolyDistribution{T}, x::T; abstol::T=zero(T), reltol::T=T(1e-6)) where {T<:Real}
   #=
   # Convergence is determined by b_i - a_i for x in (a_i, b_i).
   # (a --------- aᵢ - x - bᵢ ---- b) =>
   # Require that |b_i - a_i| < |b - a| * reltol + abstol
   # =#
   ds = domain(f.f).domains
   a = first(ds).a
   b = last(ds).b
   for d in ds
      if x in d
         #=
         # There should perhaps be a considation of what proportion of the total probability (ai, bi) * f.y[n] covers compared to p?
         =#
         return abs(d.b - d.a) < abs(b - a) * reltol + abstol
      end
   end
end

function update!(f::SparsePolyDistribution{T}, p::T, x::T, z::Bool) where {T<:Real}
   #=
   Update step:
   	if Z1 == true
     		f1(y >= x1) = inv(gamma(x1)) * p * f0(y)
     		f1(y < x1) = inv(gamma(x1)) * (1 - p) * f0(y)
     	elseif Z1 == false
     		f1(y >= x1) = inv(1 - gamma(x1)) * (1 - p) * f0(y)
     		f1(y < x1) = inv(1 - gamma(x1)) * p * f0(y)
     	end
   where
     	gamma(x) = (1 - F0(x)) * p + F0(x) * (1 - p)
   and
     	F0(x) = cumsum(f0(x))
   =#

   F::SparsePolyCumulativeDistribution{T} = cumsum(f)
   Γ::T = (one(T) - F(x)) * p + F(x) * (one(T) - p)
   if z
      # y >= x, y < x
      f′ = (inv(Γ) * p, inv(one(T) - Γ) * (one(T) - p))
   else
      # y >= x, y < x
      f′ = (inv(Γ) * (one(T) - p), inv(one(T) - Γ) * p)
   end
   f = splitDomainAt(x, f)
   _f = Fun(y -> (y >= x ? f′[1] : f′[2]), domain(f.f))
   _f = _f * f.f
   return SparsePolyDistribution(_f) # will be normalized when SparsePolyDistribution is formed
end

end
