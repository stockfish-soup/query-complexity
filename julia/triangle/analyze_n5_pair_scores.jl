include(joinpath(@__DIR__, "run_triangle_symmetric.jl"))

function graph_summary(nv::Int, rep::Integer)
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
    triangles = triangle_count_of_graph(nv, edge_index, rep)
    completing = output == 0 ? length(completing_edge_set(nv, present)) : 0
    return (
        output = output,
        edges = length(edges),
        triangles = triangles,
        completing = completing,
        degrees = sort!(degrees; rev = true),
        edge_list = edges,
    )
end

function pair_label(nv::Int, x::Integer, y::Integer)
    sx = graph_summary(nv, x)
    sy = graph_summary(nv, y)
    neg = sx.output == 0 ? sx : sy
    pos = sx.output == 1 ? sx : sy
    if neg.completing > 0 && pos.triangles == 1
        return "boundary negative to single-triangle positive"
    elseif neg.completing > 0
        return "boundary negative to multi-triangle positive"
    else
        return "interior negative to positive"
    end
end

function analyze_n5_pair_scores(; solver::Symbol = :mosek, verbose::Bool = false, time_limit_sec = nothing)
    nv = 5
    reps = full_graph_orbit_reps(nv)
    result = triangle_adversary_dual_restricted(
        nv;
        solver = solver,
        verbose = verbose,
        explicit_orbit_reps = reps,
        collect_pair_diagnostics = true,
        time_limit_sec = time_limit_sec,
    )
    println("FULL_PAIR_ANALYSIS n=$nv value=$(result.value) status=$(result.status) pair_orbits=$(result.n_pair_orbits)")

    pairs = sort(result.pair_diagnostics; by = p -> p.dual_weight, rev = true)
    total_mass = sum(p.dual_weight for p in pairs)
    mass_h1 = sum(p.dual_weight for p in pairs if p.hamming == 1)
    mass_gt1 = sum(p.dual_weight for p in pairs if p.hamming > 1)

    println("\nPair dual-weight mass by Hamming distance")
    println("distance pair_orbits mass fraction")
    for h in sort(unique(p.hamming for p in pairs))
        mass = sum(p.dual_weight for p in pairs if p.hamming == h)
        n_pairs_at_distance = count(p -> p.hamming == h, pairs)
        frac = total_mass == 0 ? 0.0 : mass / total_mass
        println("$h $n_pairs_at_distance $(round(mass, sigdigits=8)) $(round(frac, sigdigits=8))")
    end
    println("TOTAL_MASS $(round(total_mass, sigdigits=8))")
    println("HAMMING_1_MASS $(round(mass_h1, sigdigits=8)) fraction=$(round(mass_h1 / total_mass, sigdigits=8))")
    println("HAMMING_GT1_MASS $(round(mass_gt1, sigdigits=8)) fraction=$(round(mass_gt1 / total_mass, sigdigits=8))")

    println("\nMost important crossing pair orbits")
    println("rank pair_orbit weight hamming xrep yrep x_f y_f x_edges y_edges x_tri y_tri x_c y_c x_deg y_deg label")
    for (rank, p) in enumerate(Iterators.take(pairs, 40))
        sx = graph_summary(nv, p.full_x)
        sy = graph_summary(nv, p.full_y)
        println(
            "$rank $(p.pair_orbit) $(round(p.dual_weight, sigdigits=8)) $(p.hamming) " *
            "$(p.graph_rep_x) $(p.graph_rep_y) $(p.output_x) $(p.output_y) " *
            "$(sx.edges) $(sy.edges) $(sx.triangles) $(sy.triangles) " *
            "$(sx.completing) $(sy.completing) $(join(sx.degrees, ':')) $(join(sy.degrees, ':')) " *
            "\"$(pair_label(nv, p.full_x, p.full_y))\"",
        )
    end

    println("\nNonzero pair orbits with Hamming distance > 1")
    println("rank pair_orbit weight hamming xrep yrep x_edges y_edges x_tri y_tri x_c y_c label")
    gt = [p for p in pairs if p.hamming > 1 && p.dual_weight > 1e-9]
    for (rank, p) in enumerate(Iterators.take(gt, 80))
        sx = graph_summary(nv, p.full_x)
        sy = graph_summary(nv, p.full_y)
        println(
            "$rank $(p.pair_orbit) $(round(p.dual_weight, sigdigits=8)) $(p.hamming) " *
            "$(p.graph_rep_x) $(p.graph_rep_y) $(sx.edges) $(sy.edges) " *
            "$(sx.triangles) $(sy.triangles) $(sx.completing) $(sy.completing) " *
            "\"$(pair_label(nv, p.full_x, p.full_y))\"",
        )
    end

    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    solver = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :mosek
    verbose = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : false
    time_limit_sec = length(ARGS) >= 3 ? parse_optional_time_limit(ARGS[3]) : nothing
    analyze_n5_pair_scores(solver = solver, verbose = verbose, time_limit_sec = time_limit_sec)
end
