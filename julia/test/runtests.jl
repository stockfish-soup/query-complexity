using Test
include(joinpath(@__DIR__, "..", "triangle", "triangle_cb_degree_sdp.jl"))

@testset "Triangle predicate and symmetry-reduced feasibility" begin
    f, edges, _ = triangle_function(3)
    @test length(edges) == 3
    @test f == [0, 0, 0, 0, 0, 0, 0, 1]
    # Three exact edge queries give a feasible tensor of order six.
    constant = triangle_cb_degree_feasibility(3, 0; epsilon=0.0, solver=:clarabel, verbose=false)
    @test !constant.feasible
    exact = triangle_cb_degree_feasibility(3, 6; epsilon=0.0, solver=:clarabel, verbose=false)
    @test exact.feasible
end

module RestrictedAdversaryChecks
using Test
include(joinpath(@__DIR__, "..", "triangle", "triangle_adversary_dual_restricted.jl"))
@testset "SymbolicWedderburn restricted adversary" begin
    result = triangle_adversary_dual_restricted(3; solver=:clarabel, verbose=false,
        explicit_selected_full=collect(1:8))
    @test result.localN == 8
    @test isapprox(result.value, sqrt(3); atol=1e-5)
    @test !isempty(result.blocks)
end
end
