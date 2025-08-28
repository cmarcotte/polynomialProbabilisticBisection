using PPBA, Test
import PPBA: 𝒟, 𝒮, ℱ
import ApproxFunOrthogonalPolynomials: Chebyshev
import ApproxFunBase: Interval, Segment, PiecewiseSpace, roots, domain, space, sum, cumsum, Fun, ncoefficients
import DomainSets: UnionDomain

@testset "Segmented Fun Tests" begin
   for T in (Float32, Float64,)
      let a::T = zero(T), b::T = one(T), r::T = a + (b - a) * rand(T), R::T = rand(T)
         f = Fun(x -> one(T), union(Segment(a, r), Segment(r, b)))
         @test f isa ℱ
         @test sum(f) ≈ one(T)
         f = Fun(x -> (x <= r ? one(T) : one(T)), union(Segment(a, r), Segment(r, b)))
         @test f isa ℱ
         @test sum(f) ≈ one(T)
         f = Fun(x -> (x <= r ? one(T) : R), union(Segment(a, r), Segment(r, b)))
         @test f isa ℱ
         @test sum(f) ≈ (b - r) * R + (r - a) * one(T)
         PPBA.normalize!(f)
         @test sum(f) ≈ one(T)
         f = Fun(x -> (x <= r ? one(T) : zero(T)), union(Segment(a, r), Segment(r, b)))
         @test f isa ℱ
         @test sum(f) ≈ (b - r) * zero(T) + (r - a) * one(T)
         PPBA.normalize!(f)
         @test sum(f) ≈ one(T)
         @testset "Alternative formulations:" begin
            _f(x) = (x < r ? one(T) - T(1 // 2) * x + T(1 // 4) * x^2 : R)
            _F(x) = x - T(1 // 4) * x^2 + T(1 // 12) * x^3
            integrand = R * (one(T) - r) + _F(r) - _F(a)
            f0 = Fun(x -> _f(x), Interval(a, b))
            f1 = Fun(x -> _f(x), union(Interval(a, r), Interval(r, b)))
            f2 = Fun(x -> _f(x), union(Segment(a, r), Interval(r, b)))
            f3 = Fun(x -> _f(x), union(Segment(a, r), Segment(r, b)))
            f4 = Fun(x -> _f(x), union(Interval(a, r), Interval(r, b)))
            @testset "Checking Equivalencies:" begin
               @test (f0 == f1)
               @test (f0 != f2)
               @test (f2 != f3)
               @test (f3 != f4)
               @test (f1 == f4)
            end
            @testset "Checking Types:" begin
               @test !isa(f0, ℱ)
               @test !isa(f1, ℱ)
               @test !isa(f2, ℱ)
               @test isa(f3, ℱ)
               @test !isa(f4, ℱ)
            end
            @testset "Checking Accuracy:" begin
               @test !isapprox(integrand, sum(f0); atol=eps(T), rtol=zero(T))
               @test !isapprox(integrand, sum(f1); atol=eps(T), rtol=zero(T))
               @test isapprox(integrand, sum(f2); atol=eps(T), rtol=zero(T))
               @test isapprox(integrand, sum(f3); atol=eps(T), rtol=zero(T))
               @test !isapprox(integrand, sum(f4); atol=eps(T), rtol=zero(T))
            end
            @testset "Checking Coefficients:" begin
               @test ncoefficients(f0) > 5
               @test ncoefficients(f1) > 5
               @test ncoefficients(f2) == 5
               @test ncoefficients(f3) == 5
               @test ncoefficients(f4) > 5
            end
         end
      end
   end
end
@testset "SparsePolyDistribution Tests" begin
   @testset "SparsePolyDistribution Constructor Tests" begin
      for T in (Float32, Float64,)
         let a::T = zero(T), b::T = one(T), r::T = a + (b - a) * rand(T), R = rand(T)
            @testset "Fun constructor:" begin
               f = Fun(x -> x <= r ? one(T) : R, union(Segment(a, r), Segment(r, b)))
               𝒻 = SparsePolyDistribution(f)
               @test 𝒻 isa SparsePolyDistribution{T}
               @test 𝒻(R) ≈ f(R)
               @test sum(𝒻) ≈ one(T)
            end
            @testset "NTuple{N,T} constructor:" begin
               𝒻 = SparsePolyDistribution((a, b))
               @test 𝒻 isa SparsePolyDistribution{T}
               @test 𝒻(R) ≈ one(T) / (b - a)
               @test sum(𝒻) ≈ one(T)
               𝒻 = SparsePolyDistribution((a, r, b))
               @test 𝒻 isa SparsePolyDistribution{T}
               @test 𝒻(R) ≈ one(T) / (b - a)
               @test sum(𝒻) ≈ one(T)
               𝒻 = SparsePolyDistribution([a, b])
               @test 𝒻 isa SparsePolyDistribution{T}
               @test 𝒻(R) ≈ one(T) / (b - a)
               @test sum(𝒻) ≈ one(T)
               𝒻 = SparsePolyDistribution([a, r, b])
               @test 𝒻 isa SparsePolyDistribution{T}
               @test 𝒻(R) ≈ one(T) / (b - a)
               @test sum(𝒻) ≈ one(T)
            end
         end
      end
   end
   @testset "SparsePolyDistribution Evaluation Tests" begin
      for T in (Float32, Float64,)
         let a::T = zero(T), b::T = one(T), r::T = a + (b - a) * rand(T), R = rand(T)
            f = Fun(x -> x <= r ? one(T) : R, union(Segment(a, r), Segment(r, b)))
            𝒻 = SparsePolyDistribution(f)
            @test 𝒻(R) ≈ f(R)
         end
      end
   end
   @testset "Median Tests" begin
      for T in (Float32, Float64,)
         let a = zero(T), b = one(T), r::T = a + (b - a) * rand(T)

            𝒻 = SparsePolyDistribution((a, r, b))
            @test PPBA.median(𝒻) ≈ T(1 // 2) * (a + b)

            𝒻 = SparsePolyDistribution(Fun(x -> x <= r ? one(T) : zero(T), union(Segment(a, r), Segment(r, b))))
            @test PPBA.median(𝒻) ≈ T(1 // 2) * (a + r)

            𝒻 = SparsePolyDistribution(Fun(x -> x <= r ? zero(T) : one(T), union(Segment(a, r), Segment(r, b))))
            @test PPBA.median(𝒻) ≈ T(1 // 2) * (r + b)

            let s1::T = T(10) * rand(T), s2::T = T(2) * rand(T)
               nrm = (s2 * (b - r) + s1 * (r - a))
               s1′ = s1 / nrm
               s2′ = s2 / nrm
               𝒻 = SparsePolyDistribution(Fun(x -> x >= r ? s2 : s1, union(Segment(a, r), Segment(r, b))))
               @test 𝒻(T(1 // 2) * (a + r)) ≈ s1′
               @test 𝒻(T(1 // 2) * (r + b)) ≈ s2′
               @test PPBA.median(𝒻) ≈ (s1′ * r >= T(1 // 2) ? one(T) / (T(2) * s1′) : r + ((T(1 // 2) - s1′ * r) / (s2′)))
            end
         end
      end
   end
end
@testset "SparsePolyCumulativeDistribution Tests" begin
   @testset "SparsePolyCumulativeDistribution Evaluation Tests" begin
      for T in (Float32, Float64,)
         let a = zero(T), b = one(T), r::T = a + (b - a) * rand(T), s::T = a + (b - a) * rand(T)
            𝒻 = SparsePolyDistribution((a, b))
            F = PPBA.cumsum(𝒻)
            @test F isa SparsePolyCumulativeDistribution{T}
            @test F(s) ≈ 𝒻(s) * T(s - a) / (b - a)
            @test F(b) - F(a) ≈ one(T)
         end
         let a = rand(Interval(zero(T), T(1 // 2)), T), b = rand(Interval(T(1 // 2), one(T)), T), r::T = rand(Interval(a, b), T), s::T = a + (b - a) * rand(T)
            𝒻 = SparsePolyDistribution(Fun(x -> x >= r ? s : one(T), union(Segment(a, r), Segment(r, b))))
            F = PPBA.cumsum(𝒻)
            @test F isa SparsePolyCumulativeDistribution{T}
            a′ = rand(Interval(a, r), T)
            b′ = rand(Interval(r, b), T)
            @test F(a′) ≈ 𝒻(a′) * (a′ - a)
            @test F(b′) ≈ F(r) + 𝒻(b′) * (b′ - r)
            @test F(b) - F(a) ≈ one(T)
         end
      end
   end
end

@testset "PPBA Tests" begin
   @testset "Deterministic PPBA Tests" begin
      for T in (Float32, Float64,)
         let a = zero(T), b = one(T), r::T = a + (b - a) * rand(T), p = one(T)
            function Z(x::T; r::T=r, p::T=p) where {T}
               return (rand(T) <= p ? r >= x : !(r >= x))
            end
            𝒻 = SparsePolyDistribution(T[a, r, b])
            q = PPBA.median(𝒻)
            @test !(q ≈ r)
            @test !PPBA.converged(𝒻, q; reltol=T(2.0^-3), abstol=zero(T))
            for n in 1:6
               x = PPBA.median(𝒻)
               z = Z(x)
               𝒻 = PPBA.update!(𝒻, p, x, z)
            end
            q = PPBA.median(𝒻)
            @test isapprox(q, r; atol=zero(T), rtol=T(2.0^-3))
            @test PPBA.converged(𝒻, q; reltol=T(2.0^-3), abstol=zero(T))
            𝒻 = SparsePolyDistribution(T[a, b])
            q, 𝒻′ = PPBA.bisection!(Z, 𝒻, p; reltol=one(T), abstol=zero(T))
            @test isapprox(q, r; rtol=one(T), atol=zero(T))
         end
      end
   end
   @testset "Stochastic PBA Test" begin
      for T in (Float32, Float64,)
         let a = zero(T), b = one(T), r::T = a + (b - a) * rand(T), p = one(T) - T(1 // 8)
            function Z(x::T; r::T=r, p::T=p) where {T}
               return (rand(T) <= p ? r >= x : !(r >= x))
            end
            𝒻 = SparsePolyDistribution(T[a, b])
            q = PPBA.median(𝒻)
            @test !(q ≈ r)
            @test !PPBA.converged(𝒻, q; reltol=T(2.0^-3), abstol=zero(T))
            for n in 1:6
               x = PPBA.median(𝒻)
               z = Z(x)
               𝒻 = PPBA.update!(𝒻, p, x, z)
            end
            q = PPBA.median(𝒻)
            @test isapprox(q, r; atol=zero(T), rtol=T(2.0^-3))
            @test PPBA.converged(𝒻, q; reltol=T(2.0^-3), abstol=zero(T))
            𝒻 = SparsePolyDistribution(T[a, b])
            q, 𝒻′ = PPBA.bisection!(Z, 𝒻, p; reltol=one(T), abstol=zero(T))
            @test isapprox(q, r; rtol=one(T), atol=zero(T))
         end
      end
   end
end
