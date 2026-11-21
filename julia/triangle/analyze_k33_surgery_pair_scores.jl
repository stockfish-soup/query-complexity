include(joinpath(@__DIR__, "run_triangle_symmetric.jl"))

function graph_degree_sequence(nv::Int, rep::Integer)
    degrees = zeros(Int, nv)
    for (u, v) in graph_edges(nv, rep)
        degrees[u] += 1
        degrees[v] += 1
    end
    return sort!(degrees; rev = true)
end

function expected_t_surgery_sensitivity(nv::Int, a::Int, t::Int)
    b = nv - a
    return binomial(nv - 1, 2) - (a - 1) * (b - 1) + t * (a - 1)
end

function k_ab_t_surgery_reps_with_positives(
    nv::Int,
    a::Int,
    t::Int;
    include_base::Bool = true,
)
    b = nv - a
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    reps = BigInt[big(1), complete_graph_rep(nv)]
    negatives = BigInt[]
    rows = NamedTuple[]

    if include_base
        base = canonical_graph_rep(nv, complete_bipartite_rep(nv, a))
        push!(negatives, base)
        push!(reps, base)
        pos = BigInt[
            canonical_graph_rep(nv, add_edge_rep(base, edge_idx))
            for edge_idx in completing_edge_indices(nv, base, E, edge_index)
        ]
        append!(reps, pos)
        push!(rows, (
            kind = "K_{$a,$b}",
            rep = base,
            edges = count_ones(base - 1),
            completing = triangle_completing_edges(nv, base, E, edge_index),
            expected = binomial(a, 2) + binomial(b, 2),
            positives = sort!(unique!(pos)),
            degrees = graph_degree_sequence(nv, base),
        ))
    end

    expected = expected_t_surgery_sensitivity(nv, a, t)
    for surg in t_surgery_negative_reps(nv, a, t)
        push!(negatives, surg)
        push!(reps, surg)
        pos = BigInt[
            canonical_graph_rep(nv, add_edge_rep(surg, edge_idx))
            for edge_idx in completing_edge_indices(nv, surg, E, edge_index)
        ]
        append!(reps, pos)
        c = triangle_completing_edges(nv, surg, E, edge_index)
        c == expected || error(
            "bad t-surgery sensitivity for rep=$surg: got c(G)=$c, expected $expected",
        )
        push!(rows, (
            kind = "$t-surgery(K_{$a,$b})",
            rep = surg,
            edges = count_ones(surg - 1),
            completing = c,
            expected = expected,
            positives = sort!(unique!(pos)),
            degrees = graph_degree_sequence(nv, surg),
        ))
    end

    return sort!(unique!(reps)), sort!(unique!(negatives)), rows
end

function pair_graph_summary(nv::Int, rep::Integer)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    edges = graph_edges(nv, rep)
    present = Set(edges)
    output = graph_has_triangle(nv, edge_index, rep) ? 1 : 0
    return (
        output = output,
        edges = length(edges),
        triangles = triangle_count_of_graph(nv, edge_index, rep),
        completing = output == 0 ? length(completing_edge_set(nv, present)) : 0,
        degrees = graph_degree_sequence(nv, rep),
    )
end

function analyze_k33_surgery_pair_scores(;
    t::Int = 1,
    include_base::Bool = true,
    solver::Symbol = :mosek,
    verbose::Bool = false,
    time_limit_sec = nothing,
)
    nv = 6
    a = 3
    b = 3
    reps, negatives, rows = k_ab_t_surgery_reps_with_positives(
        nv,
        a,
        t;
        include_base = include_base,
    )

    println("K_{$a,$b} t-surgery construction, t=$t include_base=$include_base")
    println("expected surgery S(G) = $(expected_t_surgery_sensitivity(nv, a, t))")
    println("negative constructions:")
    for row in rows
        println(
            "  $(row.kind) rep=$(row.rep) edges=$(row.edges) " *
            "c=$(row.completing) expected=$(row.expected) " *
            "degrees=$(join(row.degrees, ':')) positives=$(row.positives)",
        )
    end
    println("total reps=$(length(reps)): $reps")

    result = triangle_adversary_dual_restricted(
        nv;
        solver = solver,
        verbose = verbose,
        explicit_orbit_reps = reps,
        collect_pair_diagnostics = true,
        time_limit_sec = time_limit_sec,
    )

    pairs = sort(result.pair_diagnostics; by = p -> p.dual_weight, rev = true)
    total_mass = sum(p.dual_weight for p in pairs)

    println(
        "\nK33_SURGERY_PAIR_ANALYSIS t=$t include_base=$include_base " *
        "value=$(result.value) status=$(result.status) inputs=$(result.localN) " *
        "pair_orbits=$(result.n_pair_orbits) total_dual_mass=$(round(total_mass, sigdigits=8))",
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
    println("HAMMING_1_MASS $(round(h1, sigdigits=8)) fraction=$(round(h1 / total_mass, sigdigits=8))")
    println("HAMMING_GT1_MASS $(round(hgt1, sigdigits=8)) fraction=$(round(hgt1 / total_mass, sigdigits=8))")

    println("\nMost important crossing pair orbits")
    println("rank pair_orbit weight hamming xrep yrep x_f y_f x_edges y_edges x_tri y_tri x_c y_c x_deg y_deg")
    for (rank, p) in enumerate(Iterators.take(pairs, 60))
        sx = pair_graph_summary(nv, p.full_x)
        sy = pair_graph_summary(nv, p.full_y)
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
    t = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
    include_base = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : true
    solver = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :mosek
    verbose = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : false
    time_limit_sec = length(ARGS) >= 5 ? parse_optional_time_limit(ARGS[5]) : nothing
    analyze_k33_surgery_pair_scores(
        t = t,
        include_base = include_base,
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
end
