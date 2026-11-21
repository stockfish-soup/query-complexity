include(joinpath(@__DIR__, "run_balanced_complete_batch.jl"))

function balanced_pair_graph_summary(nv::Int, rep::Integer)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    edges = graph_edges(nv, rep)
    present = Set(edges)
    degrees = zeros(Int, nv)
    for (u, v) in edges
        degrees[u] += 1
        degrees[v] += 1
    end
    output = graph_has_triangle(nv, edge_index, rep) ? 1 : 0
    return (
        output = output,
        edges = length(edges),
        triangles = triangle_count_of_graph(nv, edge_index, rep),
        completing = output == 0 ? length(completing_edge_set(nv, present)) : 0,
        degrees = sort!(degrees; rev = true),
    )
end

function analyze_balanced_complete_pair_scores(
    nv::Int;
    solver::Symbol = :mosek,
    verbose::Bool = false,
    time_limit_sec = nothing,
)
    selected = balanced_complete_selected_inputs(nv)
    result = triangle_adversary_dual_restricted(
        nv;
        solver = solver,
        verbose = verbose,
        explicit_selected_full = selected,
        collect_pair_diagnostics = true,
        time_limit_sec = time_limit_sec,
    )

    pairs = sort(result.pair_diagnostics; by = p -> p.dual_weight, rev = true)
    total_mass = sum(p.dual_weight for p in pairs)

    println(
        "BALANCED_COMPLETE_PAIR_ANALYSIS n=$nv split=$(fld(nv, 2)),$(nv - fld(nv, 2)) " *
        "value=$(result.value) status=$(result.status) inputs=$(result.localN) " *
        "pair_orbits=$(result.n_pair_orbits)",
    )

    println("\nPair dual-weight mass by Hamming distance")
    println("distance pair_orbits mass fraction")
    for h in sort(unique(p.hamming for p in pairs))
        mass = sum(p.dual_weight for p in pairs if p.hamming == h)
        n_pairs_at_distance = count(p -> p.hamming == h, pairs)
        frac = total_mass == 0 ? 0.0 : mass / total_mass
        println("$h $n_pairs_at_distance $(round(mass, sigdigits=8)) $(round(frac, sigdigits=8))")
    end
    h1 = sum(p.dual_weight for p in pairs if p.hamming == 1)
    hgt1 = sum(p.dual_weight for p in pairs if p.hamming > 1)
    println("TOTAL_MASS $(round(total_mass, sigdigits=8))")
    println("HAMMING_1_MASS $(round(h1, sigdigits=8)) fraction=$(round(h1 / total_mass, sigdigits=8))")
    println("HAMMING_GT1_MASS $(round(hgt1, sigdigits=8)) fraction=$(round(hgt1 / total_mass, sigdigits=8))")

    println("\nMost important crossing pair orbits")
    println("rank pair_orbit weight hamming xrep yrep x_f y_f x_edges y_edges x_tri y_tri x_c y_c x_deg y_deg")
    for (rank, p) in enumerate(Iterators.take(pairs, 40))
        sx = balanced_pair_graph_summary(nv, p.full_x)
        sy = balanced_pair_graph_summary(nv, p.full_y)
        println(
            "$rank $(p.pair_orbit) $(round(p.dual_weight, sigdigits=8)) $(p.hamming) " *
            "$(p.graph_rep_x) $(p.graph_rep_y) $(p.output_x) $(p.output_y) " *
            "$(sx.edges) $(sy.edges) $(sx.triangles) $(sy.triangles) " *
            "$(sx.completing) $(sy.completing) $(join(sx.degrees, ':')) $(join(sy.degrees, ':'))",
        )
    end
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    nv = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6
    solver = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :mosek
    verbose = length(ARGS) >= 3 ? parse(Bool, ARGS[3]) : false
    time_limit_sec = length(ARGS) >= 4 ? parse_optional_time_limit(ARGS[4]) : nothing
    analyze_balanced_complete_pair_scores(
        nv;
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
end
