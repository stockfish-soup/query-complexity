include(joinpath(@__DIR__, "triangle_adversary_dual_restricted.jl"))
using Dates
using TOML

const DEFAULT_ORBIT_STORE = joinpath(@__DIR__, "results", "triangle_orbit_results.toml")

function graph_orbits_only_direct(nv::Int, E, edge_index, fullN::Int)
    parent = collect(1:fullN)

    function find_root(x::Int)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end

    function unite!(a::Int, b::Int)
        ra = find_root(a)
        rb = find_root(b)
        ra != rb && (parent[ra] = rb)
        return nothing
    end

    vertex_gens = Vector{Int}[]
    for i in 1:(nv - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        push!(vertex_gens, sigma)
    end

    for sigma in vertex_gens, x in 1:fullN
        unite!(x, Int(apply_vertex_perm_to_graph_index(sigma, x, E, edge_index)))
    end

    sizes = Dict{Int, Int}()
    reps = Dict{Int, Int}()
    for x in 1:fullN
        r = find_root(x)
        sizes[r] = get(sizes, r, 0) + 1
        reps[r] = min(get(reps, r, x), x)
    end
    return Dict(rep => sizes[root] for (root, rep) in reps)
end

function symmetric_orbit_reps_direct(nv::Int, q::Int; prune::Bool = true)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    m = length(E)
    m < Sys.WORD_SIZE || error("direct q-threshold enumeration needs machine-size graph indices")
    fullN = 1 << m
    graph_orbit_sizes = graph_orbits_only_direct(nv, E, edge_index, fullN)

    symmetric_reps = [
        rep for (rep, orbit_size) in graph_orbit_sizes if orbit_size <= q
    ]

    reps = if prune
        complete_graph = fullN
        min_negative_edges = max(0, nv - 1)
        [
            rep for rep in symmetric_reps if
            !graph_has_triangle(nv, edge_index, rep) ?
                (rep == 1 || count_ones(rep - 1) >= min_negative_edges) :
                (rep == complete_graph || triangle_count_of_graph(nv, edge_index, rep) == 1)
        ]
    else
        symmetric_reps
    end
    sort!(reps)
    return reps
end

function symmetric_orbit_reps(nv::Int, q::Int; prune::Bool = true)
    fullN = 1 << binomial(nv, 2)
    if fullN > typemax(UInt16)
        return symmetric_orbit_reps_direct(nv, q; prune = prune)
    end

    f, E, edge_index = triangle_function(nv)
    fullN = length(f)
    Sn_gens = full_Sn_generators(nv, E, edge_index, length(E), fullN)
    graph_orbits = graph_orbits_only(fullN, Sn_gens)

    symmetric_reps = [
        rep for (rep, members) in graph_orbits if length(members) <= q
    ]

    reps = if prune
        complete_graph = fullN
        min_negative_edges = max(0, nv - 1)
        [
            rep for rep in symmetric_reps if
            f[rep] == 0 ?
                (rep == 1 || count_ones(rep - 1) >= min_negative_edges) :
                (rep == complete_graph || triangle_count_of_graph(nv, edge_index, rep) == 1)
        ]
    else
        symmetric_reps
    end
    sort!(reps)
    return reps
end

function bipartite_component_colorings(adjacency, vertices::Vector{Int}, color)
    isempty(vertices) && return [Int[]]
    start = vertices[1]
    color[start] = 1
    queue = [start]
    head = 1
    part1 = Int[start]
    while head <= length(queue)
        u = queue[head]
        head += 1
        for v in adjacency[u]
            if color[v] == 0
                color[v] = 3 - color[u]
                color[v] == 1 && push!(part1, v)
                push!(queue, v)
            elseif color[v] == color[u]
                return nothing
            end
        end
    end
    component = [v for v in 1:length(color) if color[v] != 0 && (v in queue)]
    # Isolated bookkeeping is easier if the caller passes one connected component
    # at a time; return the side containing the chosen start.
    return part1
end

function bipartite_side_sizes(nv::Int, rep::Integer, E)
    bits = rep - 1
    color = fill(0, nv)
    adjacency = [Int[] for _ in 1:nv]
    for (k, (u, v)) in enumerate(E)
        if ((bits >> (k - 1)) & 1) == 1
            push!(adjacency[u], v)
            push!(adjacency[v], u)
        end
    end

    components = Tuple{Int, Int}[]
    for start in 1:nv
        color[start] != 0 && continue
        color[start] = 1
        queue = [start]
        head = 1
        side1 = 1
        side2 = 0
        while head <= length(queue)
            u = queue[head]
            head += 1
            for v in adjacency[u]
                if color[v] == 0
                    color[v] = 3 - color[u]
                    color[v] == 1 ? (side1 += 1) : (side2 += 1)
                    push!(queue, v)
                elseif color[v] == color[u]
                    return nothing
                end
            end
        end
        push!(components, (side1, side2))
    end
    possible = Set([0])
    for (a, b) in components
        next = Set{Int}()
        for s in possible
            push!(next, s + a)
            push!(next, s + b)
        end
        possible = next
    end
    return sort!([(s, nv - s) for s in possible])
end

function is_bipartite_graph(nv::Int, rep::Integer, E)
    return bipartite_side_sizes(nv, rep, E) !== nothing
end

function has_balanced_bipartition(nv::Int, rep::Integer, E, k::Int)
    side_sizes = bipartite_side_sizes(nv, rep, E)
    side_sizes === nothing && return false
    min_side = max(0, fld(nv, 2) - k)
    return any(((a, b),) -> min(a, b) >= min_side, side_sizes)
end

function remove_edge_rep(rep::Integer, edge_idx::Int)
    return ((rep - 1) & ~(one(rep - 1) << (edge_idx - 1))) + 1
end

function is_bipartite_plus_one_positive(nv::Int, rep::Integer, E, edge_index)
    graph_has_triangle(nv, edge_index, rep) || return false
    bits = rep - 1
    for k in eachindex(E)
        ((bits >> (k - 1)) & 1) == 1 || continue
        if is_bipartite_graph(nv, remove_edge_rep(rep, k), E)
            return true
        end
    end
    return false
end

function is_balanced_bipartite_plus_one_positive(nv::Int, rep::Integer, E, edge_index, k_balance::Int)
    graph_has_triangle(nv, edge_index, rep) || return false
    bits = rep - 1
    for k in eachindex(E)
        ((bits >> (k - 1)) & 1) == 1 || continue
        if has_balanced_bipartition(nv, remove_edge_rep(rep, k), E, k_balance)
            return true
        end
    end
    return false
end

function bipartite_plus_one_orbit_reps(nv::Int; k_balance::Union{Nothing, Int} = nothing, verbose::Bool = true)
    f, E, edge_index = triangle_function(nv)
    fullN = length(f)
    Sn_gens = full_Sn_generators(nv, E, edge_index, length(E), fullN)
    graph_orbits = graph_orbits_only(fullN, Sn_gens)
    reps = Int[]
    negative_reps = Int[]
    positive_reps = Int[]
    for rep in sort(collect(keys(graph_orbits)))
        negative_ok = k_balance === nothing ?
            is_bipartite_graph(nv, rep, E) :
            has_balanced_bipartition(nv, rep, E, k_balance)
        positive_ok = k_balance === nothing ?
            is_bipartite_plus_one_positive(nv, rep, E, edge_index) :
            is_balanced_bipartite_plus_one_positive(nv, rep, E, edge_index, k_balance)
        if f[rep] == 0 && negative_ok
            push!(reps, rep)
            push!(negative_reps, rep)
        elseif f[rep] == 1 && positive_ok
            push!(reps, rep)
            push!(positive_reps, rep)
        end
    end
    sort!(unique!(reps))
    if verbose
        inputs = sum(length(graph_orbits[rep]) for rep in reps)
        descriptor = k_balance === nothing ? "bipartite-plus-one" : "balanced-bipartite-plus-one(k=$k_balance)"
        println(
            "$descriptor n=$nv: orbits=$(length(reps)) " *
            "negative_orbits=$(length(negative_reps)) positive_orbits=$(length(positive_reps)) inputs=$inputs",
        )
        println("  reps=$reps")
    end
    return reps
end

function solve_triangle_orbit_set(
    nv::Int,
    reps::AbstractVector{<:Integer};
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    m = binomial(nv, 2)
    reps = sort!(unique!(collect(reps)))
    if !(eltype(reps) <: Int) || (1 << min(m, Sys.WORD_SIZE - 1)) != (1 << m) || (1 << m) > typemax(UInt16)
        return triangle_adversary_dual_restricted(
            nv;
            solver = solver,
            verbose = verbose,
            explicit_orbit_reps = reps,
            time_limit_sec = time_limit_sec,
        )
    else
        return triangle_adversary_dual_restricted(
            nv;
            solver = solver,
            verbose = verbose,
            neg_min_edges = m + 1,
            pos_min_edges = m + 1,
            pos_max_edges = m,
            max_triangles = binomial(nv, 3),
            extra_orbit_reps = reps,
            time_limit_sec = time_limit_sec,
        )
    end
end

function solve_triangle_with_orbits(
    nv::Int,
    reps::AbstractVector{<:Integer};
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    result = solve_triangle_orbit_set(
        nv,
        sort!(unique!(collect(reps)));
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
    println("ADV+/-(triangle, n=$nv, supplied orbits) = $(result.value) [$(result.status)]")
    return result
end

function solve_triangle_with_stored_orbits(
    nv::Int,
    q::Int;
    path::AbstractString = DEFAULT_ORBIT_STORE,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    reps = stored_useful_orbits(nv, q; path = path)
    reps === nothing && error("no stored orbit set for n=$nv, q=$q in $path")
    println("Loaded $(length(reps)) useful graph orbits for n=$nv, q=$q")
    return solve_triangle_with_orbits(
        nv,
        reps;
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
end

function print_orbit_diagnostics(result; limit::Int = typemax(Int))
    diagnostics = sort(result.orbit_diagnostics; by = d -> d.importance, rev = true)
    println("\nOrbit importance (largest score first)")
    println("rank rep f orbit aut edges tri degrees obj_mult cross_mass activity score")
    for (rank, d) in enumerate(Iterators.take(diagnostics, limit))
        println(
            "$rank $(d.full_rep) $(d.output) $(d.orbit_size) $(d.automorphisms) " *
            "$(d.edges) $(d.triangles) $(join(d.degrees, ':')) " *
            "$(round(d.objective_multiplier, sigdigits=5)) " *
            "$(round(d.crossing_multiplier_mass, sigdigits=5)) " *
            "$(round(d.activity, sigdigits=5)) $(round(d.importance, sigdigits=5))",
        )
    end
    return diagnostics
end

function run_triangle_symmetric(nv::Int, q::Int; prune::Bool = true)
    reps = symmetric_orbit_reps(nv, q; prune = prune)

    result = solve_triangle_orbit_set(nv, reps; verbose = true)

    println(
        "threshold=$q value=$(result.value) status=$(result.status) " *
        "inputs=$(result.localN) pairs=$(result.n_pair_orbits) " *
        "maxblock=$(maximum(block[2] for block in result.blocks)) " *
        "variables=$(JuMP.num_variables(result.model))",
    )
    print_orbit_diagnostics(result)
    return result
end

function full_graph_orbit_reps(nv::Int)
    f, E, edge_index = triangle_function(nv)
    fullN = length(f)
    Sn_gens = full_Sn_generators(nv, E, edge_index, length(E), fullN)
    graph_orbits = graph_orbits_only(fullN, Sn_gens)
    return sort(collect(keys(graph_orbits)))
end

function run_bipartite_plus_one_comparison(nv::Int; run_full::Bool = true)
    restricted_reps = bipartite_plus_one_orbit_reps(nv; verbose = true)
    restricted = solve_triangle_with_orbits(nv, restricted_reps; verbose = true)
    println(
        "BIPARTITE_PLUS_ONE n=$nv value=$(restricted.value) " *
        "orbits=$(length(restricted_reps)) inputs=$(restricted.localN) " *
        "pairs=$(restricted.n_pair_orbits) maxblock=$(maximum(b[2] for b in restricted.blocks))",
    )
    print_orbit_diagnostics(restricted)

    full = nothing
    if run_full
        full_reps = full_graph_orbit_reps(nv)
        full = solve_triangle_with_orbits(nv, full_reps; verbose = true)
        println(
            "FULL n=$nv value=$(full.value) orbits=$(length(full_reps)) " *
            "inputs=$(full.localN) pairs=$(full.n_pair_orbits) " *
            "maxblock=$(maximum(b[2] for b in full.blocks))",
        )
        println(
            "COMPARISON n=$nv restricted=$(restricted.value) full=$(full.value) " *
            "ratio=$(restricted.value / full.value)",
        )
    end
    return (restricted = restricted, full = full, restricted_reps = restricted_reps)
end

function triangle_free_plus_one_orbit_reps(nv::Int; verbose::Bool = true)
    f, E, edge_index = triangle_function(nv)
    fullN = length(f)
    Sn_gens = full_Sn_generators(nv, E, edge_index, length(E), fullN)
    graph_orbits = graph_orbits_only(fullN, Sn_gens)

    reps = Set{Int}()
    negative_reps = Int[]
    positive_reps = Set{Int}()
    for rep in sort(collect(keys(graph_orbits)))
        f[rep] == 0 || continue
        push!(reps, rep)
        push!(negative_reps, rep)
        for edge_idx in completing_edge_indices(nv, rep, E, edge_index)
            pos = Int(canonical_graph_rep(nv, add_edge_rep(rep, edge_idx)))
            push!(reps, pos)
            push!(positive_reps, pos)
        end
    end

    verbose && println(
        "triangle-free-plus-one n=$nv: negative_orbits=$(length(negative_reps)) " *
        "positive_orbits=$(length(positive_reps)) total_orbits=$(length(reps))",
    )
    return sort!(collect(reps))
end

function run_triangle_free_plus_one_comparison(
    nv::Int;
    run_full::Bool = true,
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    restricted_reps = triangle_free_plus_one_orbit_reps(nv; verbose = true)
    restricted = solve_triangle_with_orbits(
        nv,
        restricted_reps;
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
    println(
        "TRIANGLE_FREE_PLUS_ONE n=$nv solver=$solver value=$(restricted.value) " *
        "status=$(restricted.status) orbits=$(length(restricted_reps)) " *
        "inputs=$(restricted.localN) pairs=$(restricted.n_pair_orbits) " *
        "maxblock=$(maximum(b[2] for b in restricted.blocks))",
    )

    full = nothing
    if run_full
        full_reps = full_graph_orbit_reps(nv)
        full = solve_triangle_with_orbits(
            nv,
            full_reps;
            solver = solver,
            verbose = verbose,
            time_limit_sec = time_limit_sec,
        )
        println(
            "FULL n=$nv solver=$solver value=$(full.value) status=$(full.status) " *
            "orbits=$(length(full_reps)) inputs=$(full.localN) " *
            "pairs=$(full.n_pair_orbits) maxblock=$(maximum(b[2] for b in full.blocks))",
        )
        println(
            "COMPARISON triangle-free-plus-one n=$nv restricted=$(restricted.value) " *
            "full=$(full.value) ratio=$(restricted.value / full.value) " *
            "lost=$(full.value - restricted.value)",
        )
    end
    return (restricted = restricted, full = full, restricted_reps = restricted_reps)
end

function run_triangle_full(
    nv::Int;
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    reps = full_graph_orbit_reps(nv)
    println("FULL n=$nv solver=$solver reps=$(length(reps))")
    result = solve_triangle_with_orbits(
        nv,
        reps;
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
    println(
        "RESULT full n=$nv solver=$solver value=$(result.value) " *
        "status=$(result.status) inputs=$(result.localN) " *
        "pairs=$(result.n_pair_orbits) " *
        "maxblock=$(maximum(b[2] for b in result.blocks))",
    )
    return result
end

function parse_optional_time_limit(arg::AbstractString)
    return lowercase(arg) in ("none", "nothing", "inf", "infinite", "0") ? nothing : parse(Float64, arg)
end

function run_balanced_bipartite_plus_one_comparison(nv::Int, k_balance::Int; run_full::Bool = true)
    restricted_reps = bipartite_plus_one_orbit_reps(nv; k_balance = k_balance, verbose = true)
    restricted = solve_triangle_with_orbits(nv, restricted_reps; verbose = true)
    println(
        "BALANCED_BIPARTITE_PLUS_ONE n=$nv k=$k_balance value=$(restricted.value) " *
        "orbits=$(length(restricted_reps)) inputs=$(restricted.localN) " *
        "pairs=$(restricted.n_pair_orbits) maxblock=$(maximum(b[2] for b in restricted.blocks))",
    )
    print_orbit_diagnostics(restricted)

    full = nothing
    if run_full
        full_reps = full_graph_orbit_reps(nv)
        full = solve_triangle_with_orbits(nv, full_reps; verbose = true)
        println(
            "FULL n=$nv value=$(full.value) orbits=$(length(full_reps)) " *
            "inputs=$(full.localN) pairs=$(full.n_pair_orbits) " *
            "maxblock=$(maximum(b[2] for b in full.blocks))",
        )
        println(
            "COMPARISON n=$nv k=$k_balance restricted=$(restricted.value) full=$(full.value) " *
            "ratio=$(restricted.value / full.value)",
        )
    end
    return (restricted = restricted, full = full, restricted_reps = restricted_reps)
end

function graph_rep_from_edges(nv::Int, edges::Vector{Tuple{Int, Int}})
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    bits = big(0)
    for (u, v) in edges
        bits |= big(1) << (edge_index[minmax(u, v)] - 1)
    end
    return bits + 1
end

function complete_bipartite_rep(nv::Int, a::Int)
    A = collect(1:a)
    B = collect((a + 1):nv)
    edges = Tuple{Int, Int}[]
    for u in A, v in B
        push!(edges, (u, v))
    end
    return graph_rep_from_edges(nv, edges)
end

function complete_bipartite_plus_one_rep(nv::Int, a::Int, side::Symbol)
    A = collect(1:a)
    B = collect((a + 1):nv)
    edges = Tuple{Int, Int}[]
    for u in A, v in B
        push!(edges, (u, v))
    end
    if side == :A
        length(A) >= 2 || return nothing
        push!(edges, (A[1], A[2]))
    elseif side == :B
        length(B) >= 2 || return nothing
        push!(edges, (B[1], B[2]))
    else
        error("side must be :A or :B")
    end
    return graph_rep_from_edges(nv, edges)
end

function canonical_graph_rep(nv::Int, rep::Integer)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    best = rep
    function permute!(prefix, remaining)
        if isempty(remaining)
            sigma = prefix
            image = apply_vertex_perm_to_graph_index(sigma, rep, E, edge_index)
            image < best && (best = image)
        else
            for (i, v) in enumerate(remaining)
                next_remaining = copy(remaining)
                deleteat!(next_remaining, i)
                permute!([prefix; v], next_remaining)
            end
        end
    end
    permute!(Int[], collect(1:nv))
    return best
end

function vertex_permutations(nv::Int)
    perms = Vector{Vector{Int}}()
    function permute!(prefix::Vector{Int}, remaining::Vector{Int})
        if isempty(remaining)
            push!(perms, copy(prefix))
        else
            for (i, v) in enumerate(remaining)
                push!(prefix, v)
                next_remaining = copy(remaining)
                deleteat!(next_remaining, i)
                permute!(prefix, next_remaining)
                pop!(prefix)
            end
        end
    end
    permute!(Int[], collect(1:nv))
    return perms
end

function graph_orbit_size_by_automorphisms(nv::Int, rep::Integer)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    aut = 0
    for sigma in vertex_permutations(nv)
        apply_vertex_perm_to_graph_index(sigma, rep, E, edge_index) == rep && (aut += 1)
    end
    return factorial(big(nv)) ÷ aut
end

function complete_bipartite_plus_one_orbit_reps(nv::Int; k_balance::Int = 0, verbose::Bool = true)
    min_side = max(0, fld(nv, 2) - k_balance)
    reps = Int[]
    negative = Int[]
    positive = Int[]
    for a in 0:nv
        b = nv - a
        min(a, b) >= min_side || continue
        neg = canonical_graph_rep(nv, complete_bipartite_rep(nv, a))
        push!(reps, neg)
        push!(negative, neg)
        posA = complete_bipartite_plus_one_rep(nv, a, :A)
        if posA !== nothing && b > 0
            p = canonical_graph_rep(nv, posA)
            push!(reps, p)
            push!(positive, p)
        end
        posB = complete_bipartite_plus_one_rep(nv, a, :B)
        if posB !== nothing && a > 0
            p = canonical_graph_rep(nv, posB)
            push!(reps, p)
            push!(positive, p)
        end
    end
    sort!(unique!(reps))
    sort!(unique!(negative))
    sort!(unique!(positive))
    if verbose
        inputs = sum(graph_orbit_size_by_automorphisms(nv, rep) for rep in reps)
        println(
            "complete-bipartite-plus-one n=$nv k=$k_balance: " *
            "orbits=$(length(reps)) negative_orbits=$(length(negative)) " *
            "positive_orbits=$(length(positive)) inputs=$inputs",
        )
        println("  reps=$reps")
    end
    return reps
end

function complete_graph_rep(nv::Int)
    return (big(1) << binomial(nv, 2))
end

function complete_bipartite_round_reps(nv::Int, k::Int)
    0 <= k <= fld(nv, 2) || error("k must satisfy 0 <= k <= floor(n/2)")
    reps = BigInt[big(1), complete_graph_rep(nv)]
    k == 0 && return reps

    a = k
    b = nv - k
    neg = complete_bipartite_rep(nv, a)
    push!(reps, neg)

    # For k < n/2 the two one-edge completions are not isomorphic: the extra
    # edge can lie in the small side or the large side. For k=n/2 they coincide.
    pos_small = complete_bipartite_plus_one_rep(nv, a, :A)
    pos_large = complete_bipartite_plus_one_rep(nv, a, :B)
    pos_small !== nothing && b > 0 && push!(reps, pos_small)
    pos_large !== nothing && a > 0 && push!(reps, pos_large)

    return sort!(unique!(reps))
end

function complete_bipartite_all_k_reps(nv::Int, max_k::Int = fld(nv, 2))
    0 <= max_k <= fld(nv, 2) || error("max_k must satisfy 0 <= max_k <= floor(n/2)")
    reps = BigInt[big(1), complete_graph_rep(nv)]
    for k in 1:max_k
        append!(reps, complete_bipartite_round_reps(nv, k))
    end
    return sort!(unique!(reps))
end

function surgery_graph_rep(nv::Int, a::Int, t_remove_i::Int)
    1 <= a <= fld(nv, 2) || error("a must satisfy 1 <= a <= floor(n/2)")
    b = nv - a
    b >= 2 || error("surgery graph needs |B| >= 2")
    1 <= t_remove_i <= a - 1 || error("t_remove_i must satisfy 1 <= t_remove_i <= a-1")

    A = collect(1:a)
    B = collect((a + 1):nv)
    i = B[1]
    j = B[2]
    edges = Set{Tuple{Int, Int}}()

    for u in A, v in B
        push!(edges, minmax(u, v))
    end
    push!(edges, minmax(i, j))

    # Remove one of (k,i), (k,j) for every k in A, and use both choices.
    for (pos, k) in enumerate(A)
        removed = pos <= t_remove_i ? minmax(k, i) : minmax(k, j)
        delete!(edges, removed)
    end

    return graph_rep_from_edges(nv, sort!(collect(edges)))
end

function t_surgery_graph_rep(nv::Int, a::Int, t::Int, masks::AbstractVector{Int})
    1 <= a <= fld(nv, 2) || error("a must satisfy 1 <= a <= floor(n/2)")
    b = nv - a
    1 <= t <= fld(b, 2) || error("t must satisfy 1 <= t <= floor(b/2)")
    length(masks) == t || error("one nontrivial A-partition mask is required per surgery pair")

    A = collect(1:a)
    B = collect((a + 1):nv)
    edges = Set{Tuple{Int, Int}}()
    for u in A, v in B
        push!(edges, minmax(u, v))
    end

    full_mask = (1 << a) - 1
    for r in 1:t
        u = B[2r - 1]
        v = B[2r]
        mask = masks[r]
        0 < mask < full_mask || error("each A-partition must be nontrivial")
        push!(edges, minmax(u, v))
        for (pos, x) in enumerate(A)
            in_P = ((mask >> (pos - 1)) & 1) == 1
            delete!(edges, in_P ? minmax(x, u) : minmax(x, v))
        end
    end

    return graph_rep_from_edges(nv, sort!(collect(edges)))
end

function t_surgery_negative_reps(nv::Int, a::Int, t::Int)
    b = nv - a
    1 <= t <= fld(b, 2) || return BigInt[]
    a >= 2 || return BigInt[]
    full_mask = (1 << a) - 1
    masks = collect(1:(full_mask - 1))
    reps = BigInt[]

    function rec!(chosen::Vector{Int}, depth::Int)
        if depth > t
            push!(reps, canonical_graph_rep(nv, t_surgery_graph_rep(nv, a, t, chosen)))
            return
        end
        for mask in masks
            push!(chosen, mask)
            rec!(chosen, depth + 1)
            pop!(chosen)
        end
    end
    rec!(Int[], 1)
    return sort!(unique!(reps))
end

function surgery_family_orbit_reps(
    nv::Int;
    max_a::Int = fld(nv, 2),
    max_t::Int = 1,
    verbose::Bool = true,
)
    1 <= max_a <= fld(nv, 2) || error("max_a must satisfy 1 <= max_a <= floor(n/2)")
    max_t >= 0 || error("max_t must be nonnegative")
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    reps = BigInt[big(1), complete_graph_rep(nv)]
    negative = BigInt[]
    surgery_negative = BigInt[]
    positive = BigInt[]
    rows = NamedTuple[]

    for a in 1:max_a
        b = nv - a
        cb = canonical_graph_rep(nv, complete_bipartite_rep(nv, a))
        push!(negative, cb)
        push!(reps, cb)
        cb_pos = BigInt[canonical_graph_rep(nv, add_edge_rep(cb, edge_idx)) for edge_idx in completing_edge_indices(nv, cb, E, edge_index)]
        append!(positive, cb_pos)
        append!(reps, cb_pos)
        push!(rows, (
            kind = "K_{$a,$b}",
            rep = cb,
            edges = count_ones(cb - 1),
            completing = triangle_completing_edges(nv, cb, E, edge_index),
            positives = sort!(unique!(cb_pos)),
        ))

        for t in 1:min(max_t, fld(b, 2))
            for surg in t_surgery_negative_reps(nv, a, t)
                push!(negative, surg)
                push!(surgery_negative, surg)
                push!(reps, surg)
                surg_pos = BigInt[canonical_graph_rep(nv, add_edge_rep(surg, edge_idx)) for edge_idx in completing_edge_indices(nv, surg, E, edge_index)]
                append!(positive, surg_pos)
                append!(reps, surg_pos)
                push!(rows, (
                    kind = "$(t)-surgery(a=$a,b=$b)",
                    rep = surg,
                    edges = count_ones(surg - 1),
                    completing = triangle_completing_edges(nv, surg, E, edge_index),
                    positives = sort!(unique!(surg_pos)),
                ))
            end
        end
    end

    sort!(unique!(reps))
    sort!(unique!(negative))
    sort!(unique!(surgery_negative))
    sort!(unique!(positive))

    if verbose
        println(
            "surgery-family n=$nv max_a=$max_a max_t=$max_t: orbits=$(length(reps)) " *
            "negative_orbits=$(length(negative)) surgery_negative_orbits=$(length(surgery_negative)) " *
            "positive_orbits=$(length(positive))",
        )
        println("  negative constructions:")
        for row in rows
            println(
                "    $(row.kind) rep=$(row.rep) edges=$(row.edges) " *
                "c=$(row.completing) positives=$(row.positives)",
            )
        end
        println("  reps=$reps")
    end
    return reps
end

function run_surgery_family(
    nv::Int;
    max_a::Int = fld(nv, 2),
    max_t::Int = 1,
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    reps = surgery_family_orbit_reps(nv; max_a = max_a, max_t = max_t, verbose = true)
    result = solve_triangle_with_orbits(
        nv,
        reps;
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
    println(
        "SURGERY_FAMILY n=$nv max_a=$max_a max_t=$max_t value=$(result.value) " *
        "status=$(result.status) orbits=$(length(reps)) inputs=$(result.localN) " *
        "pairs=$(result.n_pair_orbits) maxblock=$(maximum(b[2] for b in result.blocks))",
    )
    return (result = result, reps = reps)
end

function bipartite_subgraph_rep(nv::Int, a::Int, mask::Integer)
    1 <= a <= fld(nv, 2) || error("a must satisfy 1 <= a <= floor(n/2)")
    A = collect(1:a)
    B = collect((a + 1):nv)
    edges = Tuple{Int, Int}[]
    bit = 0
    for u in A, v in B
        if ((mask >> bit) & 1) == 1
            push!(edges, (u, v))
        end
        bit += 1
    end
    return graph_rep_from_edges(nv, edges)
end

function preserved_complete_bipartite_negative_reps(
    nv::Int;
    max_a::Int = fld(nv, 2),
    min_edges::Int = 0,
)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    reps = BigInt[]
    rows = NamedTuple[]
    for a in 1:max_a
        b = nv - a
        target_c = binomial(a, 2) + binomial(b, 2)
        host_edges = a * b
        host_edges <= 62 || error("host K_{$a,$b} has too many subgraphs for direct enumeration")
        local_reps = BigInt[]
        for mask in 0:((big(1) << host_edges) - 1)
            edge_count = count_ones(mask)
            edge_count >= min_edges || continue
            rep = canonical_graph_rep(nv, bipartite_subgraph_rep(nv, a, mask))
            triangle_completing_edges(nv, rep, E, edge_index) == target_c || continue
            push!(local_reps, rep)
        end
        sort!(unique!(local_reps))
        append!(reps, local_reps)
        for rep in local_reps
            push!(rows, (
                a = a,
                b = b,
                rep = rep,
                edges = count_ones(rep - 1),
                completing = triangle_completing_edges(nv, rep, E, edge_index),
                target = target_c,
            ))
        end
    end
    return sort!(unique!(reps)), rows
end

function preserved_complete_bipartite_family_reps(
    nv::Int;
    max_a::Int = fld(nv, 2),
    min_edges::Int = 0,
    verbose::Bool = true,
)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    negative, rows = preserved_complete_bipartite_negative_reps(
        nv;
        max_a = max_a,
        min_edges = min_edges,
    )
    reps = BigInt[big(1), complete_graph_rep(nv)]
    positive = BigInt[]
    append!(reps, negative)
    for neg in negative
        pos = BigInt[canonical_graph_rep(nv, add_edge_rep(neg, edge_idx)) for edge_idx in completing_edge_indices(nv, neg, E, edge_index)]
        append!(positive, pos)
        append!(reps, pos)
    end
    sort!(unique!(reps))
    sort!(unique!(positive))
    if verbose
        println(
            "preserved-bipartite n=$nv max_a=$max_a min_edges=$min_edges: " *
            "orbits=$(length(reps)) negative_orbits=$(length(negative)) " *
            "positive_orbits=$(length(positive))",
        )
        println("  negative reps preserving c(K_ab):")
        for row in sort(rows; by = r -> (r.a, -r.edges, r.rep))
            println(
                "    a=$(row.a) b=$(row.b) rep=$(row.rep) edges=$(row.edges) " *
                "c=$(row.completing) target=$(row.target)",
            )
        end
        println("  reps=$reps")
    end
    return reps, rows
end

function run_preserved_complete_bipartite_family(
    nv::Int;
    max_a::Int = fld(nv, 2),
    min_edges::Int = 0,
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    reps, rows = preserved_complete_bipartite_family_reps(
        nv;
        max_a = max_a,
        min_edges = min_edges,
        verbose = true,
    )
    result = solve_triangle_with_orbits(
        nv,
        reps;
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
    println(
        "PRESERVED_BIPARTITE n=$nv max_a=$max_a min_edges=$min_edges value=$(result.value) " *
        "status=$(result.status) orbits=$(length(reps)) negative_orbits=$(length(unique(r.rep for r in rows)) ) " *
        "inputs=$(result.localN) pairs=$(result.n_pair_orbits) maxblock=$(maximum(b[2] for b in result.blocks))",
    )
    return (result = result, reps = reps, rows = rows)
end

function preserved_host_reps_by_edge_count(nv::Int, a::Int)
    1 <= a <= fld(nv, 2) || error("a must satisfy 1 <= a <= floor(n/2)")
    b = nv - a
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    target_c = binomial(a, 2) + binomial(b, 2)
    host_edges = a * b
    host_edges <= 62 || error("host K_{$a,$b} has too many subgraphs for direct enumeration")
    by_edges = Dict{Int, Set{BigInt}}()

    for mask in 0:((big(1) << host_edges) - 1)
        rep = canonical_graph_rep(nv, bipartite_subgraph_rep(nv, a, mask))
        triangle_completing_edges(nv, rep, E, edge_index) == target_c || continue
        push!(get!(by_edges, count_ones(rep - 1), Set{BigInt}()), rep)
    end

    rows = NamedTuple[]
    for edge_count in sort(collect(keys(by_edges)); rev = true)
        for rep in sort(collect(by_edges[edge_count]))
            push!(rows, (
                rep = rep,
                edges = edge_count,
                completing = triangle_completing_edges(nv, rep, E, edge_index),
                positives = BigInt[
                    canonical_graph_rep(nv, add_edge_rep(rep, edge_idx))
                    for edge_idx in completing_edge_indices(nv, rep, E, edge_index)
                ],
            ))
        end
    end
    return rows
end

function reps_from_negative_reps(nv::Int, negative::AbstractVector{<:Integer})
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    reps = BigInt[big(1), complete_graph_rep(nv)]
    append!(reps, BigInt.(negative))
    for neg in negative
        append!(reps, BigInt[
            canonical_graph_rep(nv, add_edge_rep(neg, edge_idx))
            for edge_idx in completing_edge_indices(nv, neg, E, edge_index)
        ])
    end
    return sort!(unique!(reps))
end

function run_preserved_host_edge_sweep(
    nv::Int,
    a::Int;
    mode::Symbol = :exact,
    solver::Symbol = :mosek,
    verbose::Bool = false,
    time_limit_sec = nothing,
)
    rows = preserved_host_reps_by_edge_count(nv, a)
    edge_counts = sort!(unique([row.edges for row in rows]); rev = true)
    println("Preserved host K_{$a,$(nv-a)} n=$nv mode=$mode")
    println("target_c=$(binomial(a, 2) + binomial(nv-a, 2)) preserved_negative_orbits=$(length(rows))")
    println("edge_count negative_reps")
    for edge_count in edge_counts
        reps_here = [row.rep for row in rows if row.edges == edge_count]
        println("$edge_count $reps_here")
    end

    results = NamedTuple[]
    for edge_count in edge_counts
        negative = if mode == :exact
            [row.rep for row in rows if row.edges == edge_count]
        elseif mode == :cumulative
            [row.rep for row in rows if row.edges >= edge_count]
        else
            error("mode must be :exact or :cumulative")
        end
        reps = reps_from_negative_reps(nv, negative)
        println("\n$(uppercase(String(mode))) edge_count=$edge_count negative_orbits=$(length(negative)) total_orbits=$(length(reps))")
        result = solve_triangle_with_orbits(
            nv,
            reps;
            solver = solver,
            verbose = verbose,
            time_limit_sec = time_limit_sec,
        )
        row = (
            edge_count = edge_count,
            value = result.value,
            status = result.status,
            negative_orbits = length(negative),
            orbits = length(reps),
            inputs = result.localN,
            pair_orbits = result.n_pair_orbits,
            maxblock = maximum(b[2] for b in result.blocks),
        )
        push!(results, row)
        println(
            "PRESERVED_HOST_EDGE n=$nv a=$a mode=$mode edges=$edge_count value=$(row.value) " *
            "status=$(row.status) neg_orbits=$(row.negative_orbits) orbits=$(row.orbits) " *
            "inputs=$(row.inputs) pairs=$(row.pair_orbits) maxblock=$(row.maxblock)",
        )
    end

    println("\nSummary for K_{$a,$(nv-a)} mode=$mode")
    println("edges value status neg_orbits orbits inputs pairs maxblock")
    for row in results
        println("$(row.edge_count) $(row.value) $(row.status) $(row.negative_orbits) $(row.orbits) $(row.inputs) $(row.pair_orbits) $(row.maxblock)")
    end
    return results
end

function run_complete_bipartite_greedy(
    nv::Int;
    max_k::Int = fld(nv, 2),
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    1 <= max_k <= fld(nv, 2) || error("max_k must satisfy 1 <= max_k <= floor(n/2)")
    cumulative = BigInt[big(1), complete_graph_rep(nv)]
    results = NamedTuple[]

    println("Complete-bipartite O(n) benchmark n=$nv max_k=$max_k solver=$solver")
    for k in 1:max_k
        append!(cumulative, complete_bipartite_round_reps(nv, k))
        reps = sort!(unique!(cumulative))
        println("\nRound k=$k: complete bipartite K_{$k,$(nv-k)} family")
        println("  orbit reps=$(length(reps)): $reps")

        result = solve_triangle_with_orbits(
            nv,
            reps;
            solver = solver,
            verbose = verbose,
            time_limit_sec = time_limit_sec,
        )
        maxblock = maximum(b[2] for b in result.blocks)
        row = (
            k = k,
            value = result.value,
            status = result.status,
            orbits = length(reps),
            inputs = result.localN,
            pair_orbits = result.n_pair_orbits,
            maxblock = maxblock,
        )
        push!(results, row)
        println(
            "COMPLETE_BIPARTITE_GREEDY n=$nv k=$k value=$(row.value) " *
            "status=$(row.status) orbits=$(row.orbits) inputs=$(row.inputs) " *
            "pairs=$(row.pair_orbits) maxblock=$(row.maxblock)",
        )
    end

    println("\nComplete-bipartite benchmark summary")
    println("k value status orbits inputs pair_orbits maxblock")
    for row in results
        println(
            "$(row.k) $(row.value) $(row.status) $(row.orbits) " *
            "$(row.inputs) $(row.pair_orbits) $(row.maxblock)",
        )
    end
    return results
end

function run_complete_bipartite_plus_one_comparison(
    nv::Int,
    k_balance::Int;
    run_full::Bool = true,
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    restricted_reps = complete_bipartite_plus_one_orbit_reps(nv; k_balance = k_balance, verbose = true)
    restricted = solve_triangle_with_orbits(
        nv,
        restricted_reps;
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
    println(
        "COMPLETE_BIPARTITE_PLUS_ONE n=$nv k=$k_balance value=$(restricted.value) " *
        "orbits=$(length(restricted_reps)) inputs=$(restricted.localN) " *
        "pairs=$(restricted.n_pair_orbits) maxblock=$(maximum(b[2] for b in restricted.blocks))",
    )
    print_orbit_diagnostics(restricted)
    full = nothing
    if run_full
        full_reps = full_graph_orbit_reps(nv)
        full = solve_triangle_with_orbits(
            nv,
            full_reps;
            solver = solver,
            verbose = verbose,
            time_limit_sec = time_limit_sec,
        )
        println(
            "COMPARISON n=$nv k=$k_balance restricted=$(restricted.value) full=$(full.value) " *
            "ratio=$(restricted.value / full.value)",
        )
    end
    return (restricted = restricted, full = full, restricted_reps = restricted_reps)
end

function selected_domain_from_reps(nv::Int, reps::AbstractVector{<:Integer}; verbose::Bool = true)
    m = binomial(nv, 2)
    reps = sort!(unique!(collect(reps)))
    if (1 << min(m, Sys.WORD_SIZE - 1)) != (1 << m) || (1 << m) > typemax(UInt16)
        return explicit_orbit_domain(nv, reps; verbose = verbose)
    else
        selected, f_full, E, edge_index = select_boundary_domain(
            nv;
            neg_min_edges = m + 1,
            pos_min_edges = m + 1,
            pos_max_edges = m,
            max_triangles = binomial(nv, 3),
            extra_orbit_reps = reps,
            verbose = verbose,
        )
        return selected, f_full[selected], E, edge_index
    end
end

function restricted_generators_for_domain(nv::Int, selected_full, full_to_local, E, edge_index)
    m = length(E)
    if eltype(selected_full) <: Int && m < Sys.WORD_SIZE && (1 << m) <= typemax(UInt16)
        fullN = 1 << m
        Sn_gens = restricted_full_Sn_generators(nv, selected_full, full_to_local, E, edge_index, m, fullN)
        H_gens = restricted_stab_e0_generators(nv, selected_full, full_to_local, E, edge_index, m, fullN)
    else
        Sn_gens = restricted_full_Sn_generators_direct(nv, selected_full, full_to_local, E, edge_index)
        H_gens = restricted_stab_e0_generators_direct(nv, selected_full, full_to_local, E, edge_index)
    end
    return Sn_gens, H_gens
end

function wedderburn_blocks_for_group(H::PermGroup, localN::Int; verbose::Bool = true)
    verbose && println("  Computing Wedderburn decomposition...")
    t0 = time()
    sa_basis = SymbolicWedderburn.symmetry_adapted_basis(Float64, H; semisimple = false)
    verbose && println("  done in $(round(time() - t0, digits=2))s")

    D = isempty(sa_basis) ? 0 : size(sa_basis[1].basis, 2)
    n_extra = localN - D
    raw_blocks = []
    for i in eachindex(sa_basis)
        sz = size(sa_basis[i].basis, 1)
        if sz > 0
            rows, cols, vals = findnz(sparse(sa_basis[i].basis))
            U = sparse(rows, cols, vals, sz, localN)
            push!(raw_blocks, (
                idx = i,
                U = U,
                size = sz,
                irrep_degree = SymbolicWedderburn.degree(sa_basis[i]),
                multiplicity = SymbolicWedderburn.multiplicity(sa_basis[i]),
            ))
        end
    end
    if isempty(raw_blocks) && n_extra > 0
        push!(raw_blocks, (
            idx = 1,
            U = sparse(1.0I, localN, localN),
            size = localN,
            irrep_degree = 1,
            multiplicity = localN,
        ))
        n_extra = 0
    end
    if n_extra > 0
        trivial_idx = findfirst(blk -> any(abs(blk.U[a, 1]) > 1e-10 for a in 1:blk.size), raw_blocks)
        trivial_idx === nothing && (trivial_idx = 1)
        old = raw_blocks[trivial_idx]
        rows, cols, vals = findnz(old.U)
        for k in 1:n_extra
            push!(rows, old.size + k)
            push!(cols, D + k)
            push!(vals, 1.0)
        end
        raw_blocks[trivial_idx] = (
            idx = old.idx,
            U = sparse(rows, cols, vals, old.size + n_extra, localN),
            size = old.size + n_extra,
            irrep_degree = old.irrep_degree,
            multiplicity = old.multiplicity + n_extra,
        )
    end
    return raw_blocks
end

function graph_orbit_metadata(nv::Int, selected_full, f, E, edge_index, graph_orbits)
    graph_reps = sort(collect(keys(graph_orbits)))
    local_to_graph_orbit = zeros(Int, length(selected_full))
    metadata = NamedTuple[]
    for (gidx, rep) in enumerate(graph_reps)
        members = graph_orbits[rep]
        for x in members
            local_to_graph_orbit[x] = gidx
        end
        full_rep = minimum(selected_full[x] for x in members)
        degrees = zeros(Int, nv)
        for (j, (u, v)) in enumerate(E)
            if (((full_rep - 1) >> (j - 1)) & 1) == 1
                degrees[u] += 1
                degrees[v] += 1
            end
        end
        push!(metadata, (
            local_rep = rep,
            full_rep = full_rep,
            output = Int(f[rep]),
            orbit_size = length(members),
            automorphisms = factorial(big(nv)) ÷ length(members),
            edges = count_ones(full_rep - 1),
            triangles = triangle_count_of_graph(nv, edge_index, full_rep),
            degrees = sort!(degrees; rev = true),
        ))
    end
    return graph_reps, local_to_graph_orbit, metadata
end

function triangle_adversary_primal_restricted(
    nv::Int,
    reps::AbstractVector{<:Integer};
    solver::Symbol = :mosek,
    verbose::Bool = true,
    max_pair_members::Int = 2_000_000,
    positive_gamma::Bool = false,
)
    selected_full, f, E, edge_index = selected_domain_from_reps(nv, reps; verbose = verbose)
    localN = length(selected_full)
    full_to_local = Dict(full_x => i for (i, full_x) in enumerate(selected_full))
    verbose && println("primal restricted nv=$nv: m=$(length(E)), localN=$localN")

    count(==(Int8(0)), f) > 0 || error("restricted domain has no negative inputs")
    count(==(Int8(1)), f) > 0 || error("restricted domain has no positive inputs")

    Sn_gens, H_gens = restricted_generators_for_domain(nv, selected_full, full_to_local, E, edge_index)
    H = PermGroup([H_gens; graph_perm(1:localN)])
    H_elements = collect(H)
    verbose && println("  |H| = $(length(H_elements))")

    raw_blocks = wedderburn_blocks_for_group(H, localN; verbose = verbose)
    block_info = [(blk.idx, blk.size, blk.irrep_degree, blk.multiplicity) for blk in raw_blocks]
    if verbose
        for (idx, sz, deg, mult) in block_info
            println("    block $idx: PSD size=$sz, irrep_degree=$deg, multiplicity=$mult")
        end
    end

    verbose && println("  Computing graph and crossing-pair orbits with members...")
    t0 = time()
    graph_orbits, pair_orbits = compute_Sn_orbits(f, localN, Sn_gens; store_pair_members = true)
    total_pair_members = sum(length(p[3]) for p in pair_orbits)
    verbose && println("  done in $(round(time() - t0, digits=2))s")
    verbose && println("    graph orbits: $(length(graph_orbits))")
    verbose && println("    crossing-pair orbits: $(length(pair_orbits))")
    verbose && println("    crossing-pair members: $total_pair_members")
    total_pair_members <= max_pair_members || error(
        "primal pair-member map would contain $total_pair_members pairs; " *
        "increase max_pair_members if you really want to materialize it",
    )

    graph_reps, local_to_graph_orbit, orbit_metadata =
        graph_orbit_metadata(nv, selected_full, f, E, edge_index, graph_orbits)
    n_gorb = length(graph_reps)
    n_porb = length(pair_orbits)

    pair_index = Dict{Tuple{Int, Int}, Int}()
    pair_member_counts = zeros(Int, n_porb)
    for (pidx, (_, _, members)) in enumerate(pair_orbits)
        pair_member_counts[pidx] = length(members)
        for (x, y) in members
            f[x] == f[y] && continue
            z, o = f[x] == 0 ? (x, y) : (y, x)
            pair_index[(z, o)] = pidx
        end
    end

    optimizer = _make_restricted_optimizer(solver)
    model = Model(optimizer)
    set_silent(model)
    solver == :mosek && set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)

    @variable(model, beta[1:n_gorb] >= 0)
    if positive_gamma
        @variable(model, gamma[1:n_porb] >= 0)
    else
        @variable(model, gamma[1:n_porb])
    end
    @objective(model, Max, 2.0 * sum(pair_member_counts[p] * gamma[p] for p in 1:n_porb))
    @constraint(model, sum(orbit_metadata[i].output == 1 ? orbit_metadata[i].orbit_size * beta[i] : 0.0 for i in 1:n_gorb) == 0.5)
    @constraint(model, sum(orbit_metadata[i].output == 0 ? orbit_metadata[i].orbit_size * beta[i] : 0.0 for i in 1:n_gorb) == 0.5)

    block_cols = []
    for blk in raw_blocks
        col_rows = [Int[] for _ in 1:localN]
        col_vals = [Float64[] for _ in 1:localN]
        rows, cols, vals = findnz(blk.U)
        for k in eachindex(vals)
            push!(col_rows[cols[k]], rows[k])
            push!(col_vals[cols[k]], vals[k])
        end
        push!(block_cols, (size = blk.size, col_rows = col_rows, col_vals = col_vals))
    end

    verbose && println("  Adding block PSD constraints for canonical edge...")
    t0 = time()
    edge_bit = 1 # canonical edge {1,2} is first in edges_of_Kn
    for (bidx, blk) in enumerate(block_cols)
        M = [AffExpr(0.0) for _ in 1:blk.size, _ in 1:blk.size]

        for x in 1:localN
            rows_x = blk.col_rows[x]
            vals_x = blk.col_vals[x]
            isempty(rows_x) && continue
            bvar = beta[local_to_graph_orbit[x]]
            for aidx in eachindex(rows_x), bidx2 in eachindex(rows_x)
                add_to_expression!(
                    M[rows_x[aidx], rows_x[bidx2]],
                    vals_x[aidx] * vals_x[bidx2],
                    bvar,
                )
            end
        end

        for ((z, o), pidx) in pair_index
            (((selected_full[z] - 1) >> (edge_bit - 1)) & 1) ==
                (((selected_full[o] - 1) >> (edge_bit - 1)) & 1) && continue
            rows_z = blk.col_rows[z]
            vals_z = blk.col_vals[z]
            rows_o = blk.col_rows[o]
            vals_o = blk.col_vals[o]
            (isempty(rows_z) || isempty(rows_o)) && continue
            gvar = gamma[pidx]
            for aidx in eachindex(rows_z), bidx2 in eachindex(rows_o)
                coeff = -vals_z[aidx] * vals_o[bidx2]
                add_to_expression!(M[rows_z[aidx], rows_o[bidx2]], coeff, gvar)
                add_to_expression!(M[rows_o[bidx2], rows_z[aidx]], coeff, gvar)
            end
        end

        @constraint(model, Symmetric(M) in PSDCone())
        verbose && println("    block $bidx: PSD size=$(blk.size)")
    end
    verbose && println("  done in $(round(time() - t0, digits=2))s")

    graph_orbits = nothing
    pair_orbits = nothing
    raw_blocks = nothing
    empty!(block_cols)
    GC.gc()

    verbose && println("  Optimizing primal...")
    optimize!(model)
    status = termination_status(model)
    value = has_values(model) ? objective_value(model) : NaN

    beta_by_orbit = NamedTuple[]
    if has_values(model)
        for i in eachindex(orbit_metadata)
            b = JuMP.value(beta[i])
            push!(beta_by_orbit, merge(orbit_metadata[i], (
                beta = b,
                beta_mass = b * orbit_metadata[i].orbit_size,
            )))
        end
    end
    sort!(beta_by_orbit; by = d -> d.beta_mass, rev = true)

    verbose && println("  status: $status")
    verbose && println("  primal $(positive_gamma ? "positive " : "")ADV+/-(triangle, nv=$nv) = $value")
    return (
        value = value,
        status = status,
        localN = localN,
        n_graph_orbits = n_gorb,
        n_pair_orbits = n_porb,
        pair_members = total_pair_members,
        blocks = [(idx, sz) for (idx, sz, _, _) in block_info],
        block_details = block_info,
        beta_by_orbit = beta_by_orbit,
        model = model,
    )
end

function print_beta_diagnostics(result; limit::Int = typemax(Int))
    println("\nBeta mass by graph orbit")
    println("rank rep f orbit aut edges tri degrees beta beta_mass")
    for (rank, d) in enumerate(Iterators.take(result.beta_by_orbit, limit))
        println(
            "$rank $(d.full_rep) $(d.output) $(d.orbit_size) $(d.automorphisms) " *
            "$(d.edges) $(d.triangles) $(join(d.degrees, ':')) " *
            "$(round(d.beta, sigdigits=6)) $(round(d.beta_mass, sigdigits=6))",
        )
    end
    return result.beta_by_orbit
end

function run_triangle_primal(nv::Int, q::Int; max_pair_members::Int = 2_000_000)
    reps = symmetric_orbit_reps(nv, q; prune = true)
    result = triangle_adversary_primal_restricted(
        nv,
        reps;
        verbose = true,
        max_pair_members = max_pair_members,
    )
    println(
        "primal threshold=$q value=$(result.value) status=$(result.status) " *
        "inputs=$(result.localN) pairs=$(result.n_pair_orbits) " *
        "pair_members=$(result.pair_members) maxblock=$(maximum(block[2] for block in result.blocks)) " *
        "variables=$(JuMP.num_variables(result.model))",
    )
    print_beta_diagnostics(result)
    return result
end

orbit_store_key(nv::Int, q::Int) = "n$(nv)_q$(q)"

function load_orbit_store(path::AbstractString = DEFAULT_ORBIT_STORE)
    return isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
end

function stored_orbit_entry(nv::Int, q::Int; path::AbstractString = DEFAULT_ORBIT_STORE)
    return get(load_orbit_store(path), orbit_store_key(nv, q), nothing)
end

function stored_useful_orbits(nv::Int, q::Int; path::AbstractString = DEFAULT_ORBIT_STORE)
    entry = stored_orbit_entry(nv, q; path = path)
    entry === nothing && return nothing
    return sort!(Int.(entry["useful_orbits"]))
end

function update_orbit_store!(
    nv::Int,
    q::Int,
    reps::AbstractVector{<:Integer},
    result;
    path::AbstractString = DEFAULT_ORBIT_STORE,
    relative_tolerance::Real = 0.0,
    initial_value::Real = result.value,
    accepted_rounds::Int = 0,
)
    store = load_orbit_store(path)
    key = orbit_store_key(nv, q)
    previous = get(store, key, Dict{String, Any}())
    history = get(previous, "history", Any[])
    timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
    max_block = maximum(block[2] for block in result.blocks)
    run = Dict{String, Any}(
        "timestamp" => timestamp,
        "adversary_value" => Float64(result.value),
        "initial_value" => Float64(initial_value),
        "relative_tolerance" => Float64(relative_tolerance),
        "accepted_rounds" => accepted_rounds,
        "retained_orbits" => length(reps),
        "inputs" => result.localN,
        "pair_orbits" => result.n_pair_orbits,
        "max_psd_block" => max_block,
        "useful_orbits" => sort!(collect(Int, reps)),
    )
    push!(history, run)

    best_value = max(Float64(get(previous, "best_adversary_value", -Inf)), Float64(result.value))
    store[key] = Dict{String, Any}(
        "n" => nv,
        "q" => q,
        "updated_at" => timestamp,
        "useful_orbits" => sort!(collect(Int, reps)),
        "adversary_value" => Float64(result.value),
        "best_adversary_value" => best_value,
        "inputs" => result.localN,
        "pair_orbits" => result.n_pair_orbits,
        "max_psd_block" => max_block,
        "history" => history,
    )

    temporary = path * ".tmp"
    open(temporary, "w") do io
        TOML.print(io, store; sorted = true)
    end
    mv(temporary, path; force = true)
    println("Updated useful-orbit store: $path [$key]")
    return store[key]
end

function solve_triangle_with_stored_orbits(
    nv::Int,
    q::Int;
    path::AbstractString = DEFAULT_ORBIT_STORE,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    reps = stored_useful_orbits(nv, q; path = path)
    reps === nothing && error("no stored orbit set for n=$nv, q=$q in $path")
    println("Loaded $(length(reps)) useful graph orbits for n=$nv, q=$q")
    return solve_triangle_with_orbits(nv, reps; verbose = verbose, time_limit_sec = time_limit_sec)
end

function protected_orbit(rep::Integer, fullN::Integer)
    return rep == 1 || rep == fullN
end

function removable_candidates(result)
    diagnostics = [d for d in result.orbit_diagnostics if !protected_orbit(d.full_rep, result.fullN)]
    sort!(diagnostics; by = d -> (d.importance_per_input, d.importance, -d.orbit_size))
    return diagnostics
end

function graph_output(nv::Int, edge_index, rep::Integer)
    return graph_has_triangle(nv, edge_index, rep) ? Int8(1) : Int8(0)
end

function run_triangle_orbit_pruning(
    nv::Int,
    q::Int;
    relative_tolerance::Float64 = 1e-3,
    max_rounds::Int = 4,
    batch_fraction::Float64 = 0.20,
    store_path::AbstractString = DEFAULT_ORBIT_STORE,
    resume::Bool = true,
)
    0.0 <= relative_tolerance < 1.0 || error("relative_tolerance must lie in [0,1)")
    0.0 < batch_fraction <= 1.0 || error("batch_fraction must lie in (0,1]")

    stored_reps = resume ? stored_useful_orbits(nv, q; path = store_path) : nothing
    reps = stored_reps === nothing ? symmetric_orbit_reps(nv, q; prune = true) : stored_reps
    stored_reps !== nothing && println("Resuming from $(length(stored_reps)) stored useful graph orbits.")

    println("Solving initial orbit set with $(length(reps)) graph orbits...")
    result = solve_triangle_orbit_set(nv, reps; verbose = true)
    isfinite(result.value) || error("initial model did not return a finite objective")

    initial_value = result.value
    target_value = initial_value * (1.0 - relative_tolerance)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    println("Initial lower bound: $initial_value")
    println("Acceptance threshold: $target_value")
    print_orbit_diagnostics(result)

    accepted_rounds = 0
    for round in 1:max_rounds
        candidates = removable_candidates(result)
        isempty(candidates) && break
        requested = max(1, ceil(Int, batch_fraction * length(candidates)))
        accepted = false

        while requested >= 1
            removed = [d.full_rep for d in candidates[1:requested]]
            candidate_reps = setdiff(reps, removed)
            outputs = [graph_output(nv, edge_index, rep) for rep in candidate_reps]
            if !any(==(Int8(0)), outputs) || !any(==(Int8(1)), outputs)
                requested = fld(requested, 2)
                continue
            end

            println("\nRound $round: testing removal of $requested orbit(s): $removed")
            old_result = result
            result = nothing
            GC.gc()
            trial = solve_triangle_orbit_set(nv, candidate_reps; verbose = false)
            println(
                "  trial value=$(trial.value), inputs=$(trial.localN), " *
                "pairs=$(trial.n_pair_orbits), maxblock=$(maximum(block[2] for block in trial.blocks))",
            )

            if isfinite(trial.value) && trial.value + 1e-8 >= target_value
                reps = candidate_reps
                result = trial
                old_result = nothing
                GC.gc()
                accepted = true
                accepted_rounds += 1
                println("  accepted; rigorous restricted-domain lower bound = $(result.value)")
                print_orbit_diagnostics(result)
                update_orbit_store!(
                    nv,
                    q,
                    reps,
                    result;
                    path = store_path,
                    relative_tolerance = relative_tolerance,
                    initial_value = initial_value,
                    accepted_rounds = accepted_rounds,
                )
                break
            end

            println("  rejected; objective fell below $target_value")
            result = old_result
            trial = nothing
            GC.gc()
            requested = fld(requested, 2)
        end

        accepted || begin
            println("No acceptable removal found in round $round; stopping.")
            break
        end
    end

    if accepted_rounds == 0
        update_orbit_store!(
            nv,
            q,
            reps,
            result;
            path = store_path,
            relative_tolerance = relative_tolerance,
            initial_value = initial_value,
            accepted_rounds = 0,
        )
    end
    println("\nFinal orbit representatives: $(sort(reps))")
    println(
        "final_value=$(result.value) initial_value=$initial_value " *
        "retained_orbits=$(length(reps)) inputs=$(result.localN) accepted_rounds=$accepted_rounds",
    )
    return (result = result, reps = reps, initial_value = initial_value)
end

function triangle_completing_edges(nv::Int, full_rep::Integer, E, edge_index)
    bits = full_rep - 1
    count = 0
    for (j, (u, v)) in enumerate(E)
        ((bits >> (j - 1)) & 1) == 1 && continue
        completes_triangle = any(1:nv) do w
            (w == u || w == v) && return false
            uw = get(edge_index, minmax(u, w), 0)
            vw = get(edge_index, minmax(v, w), 0)
            return uw > 0 && vw > 0 &&
                   ((bits >> (uw - 1)) & 1) == 1 &&
                   ((bits >> (vw - 1)) & 1) == 1
        end
        count += completes_triangle
    end
    return count
end

function completing_edge_indices(nv::Int, full_rep::Integer, E, edge_index)
    bits = full_rep - 1
    indices = Int[]
    for (j, (u, v)) in enumerate(E)
        ((bits >> (j - 1)) & 1) == 1 && continue
        completes_triangle = any(1:nv) do w
            (w == u || w == v) && return false
            uw = get(edge_index, minmax(u, w), 0)
            vw = get(edge_index, minmax(v, w), 0)
            return uw > 0 && vw > 0 &&
                   ((bits >> (uw - 1)) & 1) == 1 &&
                   ((bits >> (vw - 1)) & 1) == 1
        end
        completes_triangle && push!(indices, j)
    end
    return indices
end

function add_edge_rep(rep::Integer, edge_idx::Int)
    return ((rep - 1) | (one(rep - 1) << (edge_idx - 1))) + 1
end

function orbit_metadata_by_rep(nv::Int)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    m = length(E)
    m < Sys.WORD_SIZE || error("orbit metadata scan needs machine-size graph indices")
    fullN = 1 << m
    f, _, _ = triangle_function(nv)
    orbit_sizes = graph_orbits_only_direct(nv, E, edge_index, fullN)
    meta = Dict{Int, NamedTuple}()
    for (rep, orbit_size) in orbit_sizes
        degrees = zeros(Int, nv)
        for (j, (u, v)) in enumerate(E)
            if (((rep - 1) >> (j - 1)) & 1) == 1
                degrees[u] += 1
                degrees[v] += 1
            end
        end
        tri = triangle_count_of_graph(nv, edge_index, rep)
        comp = f[rep] == 0 ? triangle_completing_edges(nv, rep, E, edge_index) : 0
        meta[rep] = (
            rep = rep,
            output = Int(f[rep]),
            orbit_size = orbit_size,
            automorphisms = factorial(big(nv)) ÷ orbit_size,
            edges = count_ones(rep - 1),
            triangles = tri,
            completing_edges = comp,
            degrees = sort!(degrees; rev = true),
        )
    end
    return meta, E, edge_index, f, fullN
end

function positive_orbits_from_negative(nv::Int, neg_rep::Integer, E, edge_index)
    reps = Int[]
    for edge_idx in completing_edge_indices(nv, neg_rep, E, edge_index)
        pos = add_edge_rep(neg_rep, edge_idx)
        push!(reps, canonical_graph_rep(nv, pos))
    end
    return sort!(unique!(reps))
end

function sorted_almost_triangle_negatives(
    nv::Int;
    min_edges::Int = 0,
    max_orbit_size::Union{Nothing, Int} = nothing,
    require_completing::Bool = true,
    ranking::Symbol = :completing,
)
    meta, E, edge_index, _, _ = orbit_metadata_by_rep(nv)
    negatives = [
        d for d in values(meta)
        if d.output == 0 &&
           d.edges >= min_edges &&
           (max_orbit_size === nothing || d.orbit_size <= max_orbit_size) &&
           (!require_completing || d.completing_edges > 0)
    ]
    if ranking == :completing
        sort!(negatives; by = d -> (-d.completing_edges, d.edges, d.orbit_size, d.rep))
    elseif ranking == :automorphism_completing
        sort!(negatives; by = d -> (-d.completing_edges, -d.automorphisms, d.edges, d.rep))
    else
        error("Unknown ranking $ranking. Use :completing or :automorphism_completing.")
    end
    return negatives, meta, E, edge_index
end

function run_almost_triangle_forward(
    nv::Int;
    rounds::Int = 8,
    min_edges::Int = 0,
    max_orbit_size::Union{Nothing, Int} = nothing,
    start_complete::Bool = true,
    solve_each_round::Bool = true,
    solver::Symbol = :mosek,
    verbose::Bool = false,
    time_limit_sec = nothing,
    ranking::Symbol = :completing,
)
    negatives, meta, E, edge_index = sorted_almost_triangle_negatives(
        nv;
        min_edges = min_edges,
        max_orbit_size = max_orbit_size,
        ranking = ranking,
    )
    fullN = 1 << binomial(nv, 2)
    reps = start_complete ? Int[1, fullN] : Int[]

    println("Almost-triangle negative candidates for n=$nv ranking=$ranking")
    println("rank rep orbit aut edges completing degrees positive_neighbors")
    for (rank, d) in enumerate(Iterators.take(negatives, max(rounds, 20)))
        pos = positive_orbits_from_negative(nv, d.rep, E, edge_index)
        println(
            "$rank $(d.rep) $(d.orbit_size) $(d.automorphisms) $(d.edges) " *
            "$(d.completing_edges) $(join(d.degrees, ':')) $pos",
        )
    end

    results = NamedTuple[]
    used_negatives = Set{Int}()
    for (round, d) in enumerate(Iterators.take(negatives, rounds))
        push!(used_negatives, d.rep)
        added = [d.rep; positive_orbits_from_negative(nv, d.rep, E, edge_index)]
        append!(reps, added)
        sort!(unique!(reps))
        println("\nRound $round: added negative $(d.rep) and positives $(setdiff(added, [d.rep]))")
        println("  orbit set has $(length(reps)) reps: $reps")
        if solve_each_round
            result = solve_triangle_with_orbits(
                nv,
                reps;
                solver = solver,
                verbose = verbose,
                time_limit_sec = time_limit_sec,
            )
            maxblock = maximum(b[2] for b in result.blocks)
            println(
                "  value=$(result.value) inputs=$(result.localN) " *
                "pairs=$(result.n_pair_orbits) maxblock=$maxblock",
            )
            push!(results, (
                round = round,
                value = result.value,
                reps = copy(reps),
                inputs = result.localN,
                pairs = result.n_pair_orbits,
                maxblock = maxblock,
            ))
            result = nothing
            GC.gc()
        end
    end
    return (results = results, reps = reps, negatives = collect(used_negatives))
end

function run_positive_almost_triangle_forward(
    nv::Int;
    rounds::Int = 8,
    min_edges::Int = 0,
    max_orbit_size::Union{Nothing, Int} = nothing,
    solver::Symbol = :mosek,
    verbose::Bool = false,
    max_pair_members::Int = 2_000_000,
    compare_dual::Bool = true,
    ranking::Symbol = :completing,
)
    negatives, meta, E, edge_index = sorted_almost_triangle_negatives(
        nv;
        min_edges = min_edges,
        max_orbit_size = max_orbit_size,
        ranking = ranking,
    )
    fullN = 1 << binomial(nv, 2)
    reps = Int[1, fullN]

    println("Positive almost-triangle greedy candidates for n=$nv ranking=$ranking")
    println("rank rep orbit aut edges completing degrees positive_neighbors")
    for (rank, d) in enumerate(Iterators.take(negatives, max(rounds, 20)))
        pos = positive_orbits_from_negative(nv, d.rep, E, edge_index)
        println(
            "$rank $(d.rep) $(d.orbit_size) $(d.automorphisms) $(d.edges) " *
            "$(d.completing_edges) $(join(d.degrees, ':')) $pos",
        )
    end

    results = NamedTuple[]
    for (round, d) in enumerate(Iterators.take(negatives, rounds))
        added = [d.rep; positive_orbits_from_negative(nv, d.rep, E, edge_index)]
        append!(reps, added)
        sort!(unique!(reps))
        println("\nRound $round: added negative $(d.rep) and positives $(setdiff(added, [d.rep]))")
        println("  orbit set has $(length(reps)) reps: $reps")

        positive_result = try
            triangle_adversary_primal_restricted(
                nv,
                reps;
                solver = solver,
                verbose = verbose,
                max_pair_members = max_pair_members,
                positive_gamma = true,
            )
        catch err
            println("  positive primal failed: $err")
            break
        end
        positive_maxblock = maximum(b[2] for b in positive_result.blocks)
        dual_value = NaN
        dual_pairs = 0
        dual_maxblock = 0
        if compare_dual
            dual_result = solve_triangle_with_orbits(
                nv,
                reps;
                solver = solver,
                verbose = verbose,
                time_limit_sec = nothing,
            )
            dual_value = dual_result.value
            dual_pairs = dual_result.n_pair_orbits
            dual_maxblock = maximum(b[2] for b in dual_result.blocks)
            dual_result = nothing
            GC.gc()
        end

        gap = isfinite(dual_value) ? dual_value - positive_result.value : NaN
        ratio = isfinite(dual_value) && dual_value != 0 ? positive_result.value / dual_value : NaN
        row = (
            round = round,
            rep = d.rep,
            edges = d.edges,
            completing = d.completing_edges,
            positive_value = positive_result.value,
            dual_value = dual_value,
            gap = gap,
            ratio = ratio,
            orbits = length(reps),
            inputs = positive_result.localN,
            pair_orbits = positive_result.n_pair_orbits,
            pair_members = positive_result.pair_members,
            positive_maxblock = positive_maxblock,
            dual_pairs = dual_pairs,
            dual_maxblock = dual_maxblock,
        )
        push!(results, row)
        println(
            "POSITIVE_FORWARD n=$nv round=$round rep=$(d.rep) edges=$(d.edges) c=$(d.completing_edges) " *
            "positive=$(row.positive_value) dual=$(row.dual_value) gap=$(row.gap) ratio=$(row.ratio) " *
            "orbits=$(row.orbits) inputs=$(row.inputs) pair_orbits=$(row.pair_orbits) " *
            "pair_members=$(row.pair_members) maxblock=$(row.positive_maxblock)",
        )
        positive_result = nothing
        GC.gc()
    end

    println("\nPositive greedy summary")
    println("round rep edges c positive dual gap ratio orbits inputs pair_orbits pair_members maxblock")
    for r in results
        println("$(r.round) $(r.rep) $(r.edges) $(r.completing) $(r.positive_value) $(r.dual_value) $(r.gap) $(r.ratio) $(r.orbits) $(r.inputs) $(r.pair_orbits) $(r.pair_members) $(r.positive_maxblock)")
    end
    return (results = results, reps = reps)
end

function run_family_comparison(nv::Int)
    println("\nComplete bipartite plus one:")
    for k in 0:min(nv, 4)
        r = run_complete_bipartite_plus_one_comparison(nv, k; run_full = false).restricted
        println("FAMILY complete_bipartite k=$k value=$(r.value) reps=$(r.n_graph_orbits) inputs=$(r.localN)")
    end
    println("\nAlmost-triangle greedy:")
    return run_almost_triangle_forward(nv; rounds = 8, min_edges = 0, max_orbit_size = nothing)
end

function forward_candidate_metadata(nv::Int, q::Int, selected::AbstractVector{<:Integer})
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    m = length(E)
    m < Sys.WORD_SIZE || error("candidate scan needs machine-size graph indices")
    fullN = 1 << m
    orbit_sizes = graph_orbits_only_direct(nv, E, edge_index, fullN)
    selected_set = Set(selected)
    metadata = NamedTuple[]

    for (rep, orbit_size) in orbit_sizes
        orbit_size <= q || continue
        rep in selected_set && continue
        degrees = zeros(Int, nv)
        for (j, (u, v)) in enumerate(E)
            if (((rep - 1) >> (j - 1)) & 1) == 1
                degrees[u] += 1
                degrees[v] += 1
            end
        end
        output = Int(graph_output(nv, edge_index, rep))
        push!(metadata, (
            rep = rep,
            output = output,
            orbit_size = orbit_size,
            automorphisms = factorial(big(nv)) ÷ orbit_size,
            edges = count_ones(rep - 1),
            triangles = triangle_count_of_graph(nv, edge_index, rep),
            completing_edges = output == 0 ? triangle_completing_edges(nv, rep, E, edge_index) : 0,
            degrees = sort!(degrees; rev = true),
        ))
    end
    return metadata
end

function screen_forward_candidates(metadata, candidate_limit::Int)
    candidate_limit > 0 || return NamedTuple[]
    negatives = sort(
        [d for d in metadata if d.output == 0];
        by = d -> (-d.completing_edges, d.edges, d.orbit_size, d.rep),
    )
    positives = sort(
        [d for d in metadata if d.output == 1];
        by = d -> (d.triangles, d.edges, d.orbit_size, d.rep),
    )
    negative_limit = min(length(negatives), cld(candidate_limit, 2))
    positive_limit = min(length(positives), fld(candidate_limit, 2))
    screened = vcat(negatives[1:negative_limit], positives[1:positive_limit])
    chosen = Set(d.rep for d in screened)
    remainder = sort(
        [d for d in metadata if !(d.rep in chosen)];
        by = d -> (d.orbit_size, d.triangles, -d.completing_edges, d.rep),
    )
    append!(screened, Iterators.take(remainder, candidate_limit - length(screened)))
    return screened
end

function run_triangle_orbit_forward(
    nv::Int,
    q::Int;
    max_rounds::Int = 5,
    candidate_limit::Int = 12,
    minimum_gain::Float64 = 1e-5,
    resume::Bool = true,
    store_path::AbstractString = DEFAULT_ORBIT_STORE,
)
    max_rounds >= 0 || error("max_rounds must be nonnegative")
    candidate_limit > 0 || error("candidate_limit must be positive")
    minimum_gain >= 0.0 || error("minimum_gain must be nonnegative")

    fullN = 1 << binomial(nv, 2)
    stored = resume ? stored_useful_orbits(nv, q; path = store_path) : nothing
    reps = stored === nothing ? [1, fullN] : stored
    sort!(unique!(reps))
    println(stored === nothing ?
        "Starting from the empty and complete graph orbits." :
        "Resuming from $(length(reps)) stored useful graph orbits.")

    current = solve_triangle_with_orbits(nv, reps; verbose = true)
    initial_value = current.value
    isfinite(initial_value) || error("initial orbit set did not produce a finite objective")

    accepted_rounds = 0
    for round in 1:max_rounds
        metadata = forward_candidate_metadata(nv, q, reps)
        candidates = screen_forward_candidates(metadata, candidate_limit)
        isempty(candidates) && begin
            println("No unselected candidate orbits remain.")
            break
        end

        baseline = current.value
        println("\nRound $round: baseline=$baseline, testing $(length(candidates)) candidate orbits")
        for d in candidates
            println(
                "  candidate rep=$(d.rep) f=$(d.output) orbit=$(d.orbit_size) " *
                "aut=$(d.automorphisms) edges=$(d.edges) triangles=$(d.triangles) " *
                "completing_edges=$(d.completing_edges) degrees=$(d.degrees)",
            )
        end

        current = nothing
        GC.gc()
        trial_values = Tuple{Float64, NamedTuple}[]
        for d in candidates
            trial = solve_triangle_orbit_set(nv, sort([reps; d.rep]); verbose = false)
            println("  rep=$(d.rep): value=$(trial.value), gain=$(trial.value - baseline)")
            push!(trial_values, (trial.value, d))
            trial = nothing
            GC.gc()
        end

        sort!(trial_values; by = first, rev = true)
        best_value, best = first(trial_values)
        gain = best_value - baseline
        if !isfinite(best_value) || gain + 1e-8 < minimum_gain
            println("Best marginal gain $gain is below minimum_gain=$minimum_gain; stopping.")
            current = solve_triangle_orbit_set(nv, reps; verbose = false)
            break
        end

        push!(reps, best.rep)
        sort!(reps)
        println("Adding orbit $(best.rep), with certified trial value $best_value and gain $gain")
        current = solve_triangle_orbit_set(nv, reps; verbose = false)
        accepted_rounds += 1
        update_orbit_store!(
            nv,
            q,
            reps,
            current;
            path = store_path,
            initial_value = initial_value,
            accepted_rounds = accepted_rounds,
        )
    end

    accepted_rounds == 0 && update_orbit_store!(
        nv,
        q,
        reps,
        current;
        path = store_path,
        initial_value = initial_value,
        accepted_rounds = 0,
    )
    println("\nUseful orbit representatives: $(sort(reps))")
    println("ADV+/-(triangle, n=$nv) = $(current.value) [$(current.status)]")
    return (result = current, reps = reps, initial_value = initial_value)
end

const N7_IMPORTANT_GRAPHS = [
    (rank = 1, rep = 3903, f = 1, orbit = 420, score = 0.991),
    (rank = 2, rep = 3965, f = 0, orbit = 420, score = 0.950),
    (rank = 3, rep = 96, f = 1, orbit = 420, score = 0.911),
    (rank = 4, rep = 128, f = 1, orbit = 105, score = 0.857),
    (rank = 5, rep = 2298, f = 1, orbit = 420, score = 0.853),
    (rank = 6, rep = 64, f = 0, orbit = 7, score = 0.733),
    (rank = 7, rep = 255, f = 0, orbit = 420, score = 0.710),
    (rank = 8, rep = 250, f = 0, orbit = 420, score = 0.696),
    (rank = 9, rep = 126, f = 0, orbit = 210, score = 0.642),
    (rank = 10, rep = 72, f = 1, orbit = 420, score = 0.552),
]

function graph_edges(nv::Int, rep::Integer)
    E = edges_of_Kn(nv)
    bits = rep - 1
    edges = Tuple{Int, Int}[]
    for (k, edge) in enumerate(E)
        ((bits >> (k - 1)) & 1) == 1 && push!(edges, edge)
    end
    return edges
end

function actual_triangle_edges(nv::Int, present::Set{Tuple{Int, Int}})
    highlighted = Set{Tuple{Int, Int}}()
    for a in 1:nv, b in (a + 1):nv, c in (b + 1):nv
        tri = (minmax(a, b), minmax(a, c), minmax(b, c))
        all(edge -> edge in present, tri) && foreach(edge -> push!(highlighted, edge), tri)
    end
    return highlighted
end

function completing_edge_set(nv::Int, present::Set{Tuple{Int, Int}})
    missing = Set{Tuple{Int, Int}}()
    for a in 1:nv, b in (a + 1):nv
        edge = minmax(a, b)
        edge in present && continue
        for c in 1:nv
            (c == a || c == b) && continue
            if minmax(a, c) in present && minmax(b, c) in present
                push!(missing, edge)
                break
            end
        end
    end
    return missing
end

function svg_line(x1, y1, x2, y2; color = "#222222", width = 2.0, dash = "")
    dash_attr = isempty(dash) ? "" : " stroke-dasharray=\"$dash\""
    return "<line x1=\"$x1\" y1=\"$y1\" x2=\"$x2\" y2=\"$y2\" stroke=\"$color\" stroke-width=\"$width\" stroke-linecap=\"round\"$dash_attr />\n"
end

function draw_graph_panel(io, g, x0, y0)
    nv = 7
    radius = 54.0
    cx = x0 + 82.0
    cy = y0 + 85.0
    coords = Dict{Int, Tuple{Float64, Float64}}()
    for v in 1:nv
        theta = -pi / 2 + 2pi * (v - 1) / nv
        coords[v] = (round(cx + radius * cos(theta), digits = 2), round(cy + radius * sin(theta), digits = 2))
    end

    edges = graph_edges(nv, g.rep)
    present = Set(minmax(u, v) for (u, v) in edges)
    red_edges = actual_triangle_edges(nv, present)
    dashed_edges = g.f == 0 ? completing_edge_set(nv, present) : Set{Tuple{Int, Int}}()

    println(io, "<g transform=\"translate($x0,$y0)\">")
    println(io, "<rect x=\"0\" y=\"0\" width=\"164\" height=\"176\" rx=\"8\" fill=\"#ffffff\" stroke=\"#d8dde6\" />")
    println(io, "<text x=\"10\" y=\"20\" font-size=\"12\" font-family=\"Inter,Arial,sans-serif\" fill=\"#0f172a\">#$(g.rank) rep $(g.rep)</text>")
    println(io, "<text x=\"10\" y=\"38\" font-size=\"11\" font-family=\"Inter,Arial,sans-serif\" fill=\"#475569\">f=$(g.f) orbit=$(g.orbit) score=$(g.score)</text>")

    for (u, v) in sort!(collect(dashed_edges))
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        print(io, svg_line(x1 - x0, y1 - y0, x2 - x0, y2 - y0; color = "#ef9a9a", width = 1.5, dash = "4 4"))
    end
    for (u, v) in edges
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        edge = minmax(u, v)
        color = edge in red_edges ? "#dc2626" : "#1f2937"
        width = edge in red_edges ? 3.2 : 2.0
        print(io, svg_line(x1 - x0, y1 - y0, x2 - x0, y2 - y0; color = color, width = width))
    end
    for v in 1:nv
        x, y = coords[v]
        println(io, "<circle cx=\"$(x - x0)\" cy=\"$(y - y0)\" r=\"10\" fill=\"#f8fafc\" stroke=\"#0f172a\" stroke-width=\"1.5\" />")
        println(io, "<text x=\"$(x - x0)\" y=\"$(y - y0 + 4)\" text-anchor=\"middle\" font-size=\"10\" font-family=\"Inter,Arial,sans-serif\" fill=\"#0f172a\">$v</text>")
    end
    println(io, "</g>")
end

function draw_n7_important_orbits(out::AbstractString = joinpath(@__DIR__, "results", "n7_important_orbits.svg"))
    open(out, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">""")
        println(io, """<rect width="900" height="500" fill="#f6f8fb" />""")
        println(io, """<text x="24" y="30" font-size="18" font-family="Inter,Arial,sans-serif" font-weight="700" fill="#0f172a">n=7 important graph orbits, q=500</text>""")
        println(io, """<text x="24" y="52" font-size="12" font-family="Inter,Arial,sans-serif" fill="#475569">Black = present edge, red = actual triangle, dashed red = missing edge that would create a triangle.</text>""")
        for (i, g) in enumerate(N7_IMPORTANT_GRAPHS)
            col = (i - 1) % 5
            row = (i - 1) ÷ 5
            draw_graph_panel(io, g, 20 + 174 * col, 72 + 198 * row)
        end
        println(io, "</svg>")
    end
    println(out)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS) && ARGS[1] == "solve"
        length(ARGS) >= 3 || error("Usage: julia --project=. run_triangle_symmetric.jl solve <n> <rep...>|--stored <q>")
        nv = parse(Int, ARGS[2])
        if ARGS[3] == "--stored"
            solve_triangle_with_stored_orbits(nv, parse(Int, ARGS[4]))
        else
            solve_triangle_with_orbits(nv, parse.(Int, ARGS[3:end]))
        end
    elseif !isempty(ARGS) && ARGS[1] == "full"
        2 <= length(ARGS) <= 5 || error("Usage: julia --project=. run_triangle_symmetric.jl full <n> [solver=mosek] [verbose=true] [time_limit_sec=7200|none|0]")
        run_triangle_full(
            parse(Int, ARGS[2]);
            solver = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :mosek,
            verbose = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : true,
            time_limit_sec = length(ARGS) >= 5 ? parse_optional_time_limit(ARGS[5]) : 7200.0,
        )
    elseif !isempty(ARGS) && ARGS[1] == "prune"
        3 <= length(ARGS) <= 6 || error("Usage: julia --project=. run_triangle_symmetric.jl prune <n> <q> [tol=0.001] [rounds=4] [fraction=0.20]")
        run_triangle_orbit_pruning(
            parse(Int, ARGS[2]),
            parse(Int, ARGS[3]);
            relative_tolerance = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-3,
            max_rounds = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 4,
            batch_fraction = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.20,
        )
    elseif !isempty(ARGS) && ARGS[1] == "forward"
        3 <= length(ARGS) <= 7 || error("Usage: julia --project=. run_triangle_symmetric.jl forward <n> <q> [rounds=5] [candidates=12] [gain=1e-5] [resume=true]")
        run_triangle_orbit_forward(
            parse(Int, ARGS[2]),
            parse(Int, ARGS[3]);
            max_rounds = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5,
            candidate_limit = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 12,
            minimum_gain = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1e-5,
            resume = length(ARGS) >= 7 ? parse(Bool, ARGS[7]) : true,
        )
    elseif !isempty(ARGS) && ARGS[1] == "primal"
        3 <= length(ARGS) <= 4 || error("Usage: julia --project=. run_triangle_symmetric.jl primal <n> <q> [max_pair_members=2000000]")
        run_triangle_primal(
            parse(Int, ARGS[2]),
            parse(Int, ARGS[3]);
            max_pair_members = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 2_000_000,
        )
    elseif !isempty(ARGS) && ARGS[1] == "bipartite-plus-one"
        2 <= length(ARGS) <= 3 || error("Usage: julia --project=. run_triangle_symmetric.jl bipartite-plus-one <n> [run_full=true]")
        run_bipartite_plus_one_comparison(
            parse(Int, ARGS[2]);
            run_full = length(ARGS) >= 3 ? parse(Bool, ARGS[3]) : true,
        )
    elseif !isempty(ARGS) && ARGS[1] == "triangle-free-plus-one"
        2 <= length(ARGS) <= 6 || error("Usage: julia --project=. run_triangle_symmetric.jl triangle-free-plus-one <n> [run_full=true] [solver=mosek] [verbose=true] [time_limit_sec=none|0|7200]")
        run_triangle_free_plus_one_comparison(
            parse(Int, ARGS[2]);
            run_full = length(ARGS) >= 3 ? parse(Bool, ARGS[3]) : true,
            solver = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :mosek,
            verbose = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : true,
            time_limit_sec = length(ARGS) >= 6 ? parse_optional_time_limit(ARGS[6]) : nothing,
        )
    elseif !isempty(ARGS) && ARGS[1] == "balanced-bipartite-plus-one"
        3 <= length(ARGS) <= 4 || error("Usage: julia --project=. run_triangle_symmetric.jl balanced-bipartite-plus-one <n> <k> [run_full=true]")
        run_balanced_bipartite_plus_one_comparison(
            parse(Int, ARGS[2]),
            parse(Int, ARGS[3]);
            run_full = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : true,
        )
    elseif !isempty(ARGS) && ARGS[1] == "complete-bipartite-plus-one"
        3 <= length(ARGS) <= 7 || error("Usage: julia --project=. run_triangle_symmetric.jl complete-bipartite-plus-one <n> <k> [run_full=true] [solver=mosek] [verbose=true] [time_limit_sec=7200|none|0]")
        run_complete_bipartite_plus_one_comparison(
            parse(Int, ARGS[2]),
            parse(Int, ARGS[3]);
            run_full = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : true,
            solver = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :mosek,
            verbose = length(ARGS) >= 6 ? parse(Bool, ARGS[6]) : true,
            time_limit_sec = length(ARGS) >= 7 ? parse_optional_time_limit(ARGS[7]) : 7200.0,
        )
    elseif !isempty(ARGS) && ARGS[1] == "complete-bipartite-greedy"
        2 <= length(ARGS) <= 6 || error("Usage: julia --project=. run_triangle_symmetric.jl complete-bipartite-greedy <n> [max_k=floor(n/2)] [solver=mosek] [verbose=true] [time_limit_sec=none|0|7200]")
        nv = parse(Int, ARGS[2])
        run_complete_bipartite_greedy(
            nv;
            max_k = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : fld(nv, 2),
            solver = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :mosek,
            verbose = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : true,
            time_limit_sec = length(ARGS) >= 6 ? parse_optional_time_limit(ARGS[6]) : nothing,
        )
    elseif !isempty(ARGS) && ARGS[1] == "surgery-family"
        2 <= length(ARGS) <= 7 || error("Usage: julia --project=. run_triangle_symmetric.jl surgery-family <n> [max_t=1] [max_a=floor(n/2)] [solver=mosek] [verbose=true] [time_limit_sec=none|0|7200]")
        nv = parse(Int, ARGS[2])
        run_surgery_family(
            nv;
            max_t = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1,
            max_a = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : fld(nv, 2),
            solver = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :mosek,
            verbose = length(ARGS) >= 6 ? parse(Bool, ARGS[6]) : true,
            time_limit_sec = length(ARGS) >= 7 ? parse_optional_time_limit(ARGS[7]) : nothing,
        )
    elseif !isempty(ARGS) && ARGS[1] == "preserved-bipartite"
        2 <= length(ARGS) <= 7 || error("Usage: julia --project=. run_triangle_symmetric.jl preserved-bipartite <n> [min_edges=0] [max_a=floor(n/2)] [solver=mosek] [verbose=true] [time_limit_sec=none|0|7200]")
        nv = parse(Int, ARGS[2])
        run_preserved_complete_bipartite_family(
            nv;
            min_edges = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0,
            max_a = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : fld(nv, 2),
            solver = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :mosek,
            verbose = length(ARGS) >= 6 ? parse(Bool, ARGS[6]) : true,
            time_limit_sec = length(ARGS) >= 7 ? parse_optional_time_limit(ARGS[7]) : nothing,
        )
    elseif !isempty(ARGS) && ARGS[1] == "preserved-host-edge-sweep"
        3 <= length(ARGS) <= 7 || error("Usage: julia --project=. run_triangle_symmetric.jl preserved-host-edge-sweep <n> <a> [mode=exact|cumulative] [solver=mosek] [verbose=false] [time_limit_sec=none|0|7200]")
        run_preserved_host_edge_sweep(
            parse(Int, ARGS[2]),
            parse(Int, ARGS[3]);
            mode = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :exact,
            solver = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :mosek,
            verbose = length(ARGS) >= 6 ? parse(Bool, ARGS[6]) : false,
            time_limit_sec = length(ARGS) >= 7 ? parse_optional_time_limit(ARGS[7]) : nothing,
        )
    elseif !isempty(ARGS) && ARGS[1] == "almost-triangle-forward"
        2 <= length(ARGS) <= 8 || error("Usage: julia --project=. run_triangle_symmetric.jl almost-triangle-forward <n> [rounds=8] [min_edges=0] [max_orbit_size=0 means none] [solver=mosek] [verbose=false] [time_limit_sec=7200|none|0]")
        max_orbit_arg = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0
        run_almost_triangle_forward(
            parse(Int, ARGS[2]);
            rounds = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8,
            min_edges = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0,
            max_orbit_size = max_orbit_arg == 0 ? nothing : max_orbit_arg,
            solver = length(ARGS) >= 6 ? Symbol(ARGS[6]) : :mosek,
            verbose = length(ARGS) >= 7 ? parse(Bool, ARGS[7]) : false,
            time_limit_sec = length(ARGS) >= 8 ? parse_optional_time_limit(ARGS[8]) : 7200.0,
        )
    elseif !isempty(ARGS) && ARGS[1] == "positive-almost-forward"
        2 <= length(ARGS) <= 9 || error("Usage: julia --project=. run_triangle_symmetric.jl positive-almost-forward <n> [rounds=8] [min_edges=0] [max_orbit_size=0 means none] [solver=mosek] [verbose=false] [max_pair_members=2000000] [compare_dual=true]")
        max_orbit_arg = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0
        run_positive_almost_triangle_forward(
            parse(Int, ARGS[2]);
            rounds = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8,
            min_edges = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0,
            max_orbit_size = max_orbit_arg == 0 ? nothing : max_orbit_arg,
            solver = length(ARGS) >= 6 ? Symbol(ARGS[6]) : :mosek,
            verbose = length(ARGS) >= 7 ? parse(Bool, ARGS[7]) : false,
            max_pair_members = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 2_000_000,
            compare_dual = length(ARGS) >= 9 ? parse(Bool, ARGS[9]) : true,
        )
    elseif !isempty(ARGS) && ARGS[1] == "auto-sg-forward"
        2 <= length(ARGS) <= 8 || error("Usage: julia --project=. run_triangle_symmetric.jl auto-sg-forward <n> [rounds=8] [min_edges=0] [max_orbit_size=0 means none] [solver=mosek] [verbose=false] [time_limit_sec=7200|none|0]")
        max_orbit_arg = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0
        run_almost_triangle_forward(
            parse(Int, ARGS[2]);
            rounds = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8,
            min_edges = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0,
            max_orbit_size = max_orbit_arg == 0 ? nothing : max_orbit_arg,
            solver = length(ARGS) >= 6 ? Symbol(ARGS[6]) : :mosek,
            verbose = length(ARGS) >= 7 ? parse(Bool, ARGS[7]) : false,
            time_limit_sec = length(ARGS) >= 8 ? parse_optional_time_limit(ARGS[8]) : 7200.0,
            ranking = :automorphism_completing,
        )
    elseif !isempty(ARGS) && ARGS[1] == "families"
        length(ARGS) == 2 || error("Usage: julia --project=. run_triangle_symmetric.jl families <n>")
        run_family_comparison(parse(Int, ARGS[2]))
    elseif !isempty(ARGS) && ARGS[1] == "draw-n7"
        draw_n7_important_orbits(length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "results", "n7_important_orbits.svg"))
    elseif length(ARGS) == 2
        run_triangle_symmetric(parse(Int, ARGS[1]), parse(Int, ARGS[2]))
    else
        error(
            "Usage:\n" *
            "  julia --project=. run_triangle_symmetric.jl <n> <q>\n" *
            "  julia --project=. run_triangle_symmetric.jl solve <n> <rep...>|--stored <q>\n" *
            "  julia --project=. run_triangle_symmetric.jl full <n> [solver] [verbose] [time_limit_sec]\n" *
            "  julia --project=. run_triangle_symmetric.jl prune <n> <q> [tol] [rounds] [fraction]\n" *
            "  julia --project=. run_triangle_symmetric.jl forward <n> <q> [rounds] [candidates] [gain] [resume]\n" *
            "  julia --project=. run_triangle_symmetric.jl primal <n> <q> [max_pair_members]\n" *
            "  julia --project=. run_triangle_symmetric.jl bipartite-plus-one <n> [run_full]\n" *
            "  julia --project=. run_triangle_symmetric.jl triangle-free-plus-one <n> [run_full] [solver] [verbose] [time_limit_sec]\n" *
            "  julia --project=. run_triangle_symmetric.jl balanced-bipartite-plus-one <n> <k> [run_full]\n" *
            "  julia --project=. run_triangle_symmetric.jl complete-bipartite-plus-one <n> <k> [run_full] [solver] [verbose] [time_limit_sec]\n" *
            "  julia --project=. run_triangle_symmetric.jl complete-bipartite-greedy <n> [max_k] [solver] [verbose] [time_limit_sec]\n" *
            "  julia --project=. run_triangle_symmetric.jl surgery-family <n> [max_t] [max_a] [solver] [verbose] [time_limit_sec]\n" *
            "  julia --project=. run_triangle_symmetric.jl preserved-bipartite <n> [min_edges] [max_a] [solver] [verbose] [time_limit_sec]\n" *
            "  julia --project=. run_triangle_symmetric.jl preserved-host-edge-sweep <n> <a> [mode] [solver] [verbose] [time_limit_sec]\n" *
            "  julia --project=. run_triangle_symmetric.jl almost-triangle-forward <n> [rounds] [min_edges] [max_orbit_size] [solver] [verbose] [time_limit_sec]\n" *
            "  julia --project=. run_triangle_symmetric.jl positive-almost-forward <n> [rounds] [min_edges] [max_orbit_size] [solver] [verbose] [max_pair_members] [compare_dual]\n" *
            "  julia --project=. run_triangle_symmetric.jl auto-sg-forward <n> [rounds] [min_edges] [max_orbit_size] [solver] [verbose] [time_limit_sec]\n" *
            "  julia --project=. run_triangle_symmetric.jl families <n>\n" *
            "  julia --project=. run_triangle_symmetric.jl draw-n7 [output.svg]"
        )
    end
end
