include(joinpath(@__DIR__, "run_triangle_symmetric.jl"))

function combinations_vec(items::Vector{Int}, k::Int)
    out = Vector{Vector{Int}}()
    current = Int[]
    function rec(start::Int, remaining::Int)
        if remaining == 0
            push!(out, copy(current))
            return
        end
        last_start = length(items) - remaining + 1
        for i in start:last_start
            push!(current, items[i])
            rec(i + 1, remaining - 1)
            pop!(current)
        end
    end
    rec(1, k)
    return out
end

function matchings_of_size(vertices::Vector{Int}, t::Int)
    t == 0 && return [Tuple{Int, Int}[]]
    out = Vector{Vector{Tuple{Int, Int}}}()
    function rec(remaining::Vector{Int}, chosen::Vector{Tuple{Int, Int}})
        if length(chosen) == t
            push!(out, copy(chosen))
            return
        end
        length(remaining) < 2 * (t - length(chosen)) && return
        u = remaining[1]
        rest = remaining[2:end]
        for (idx, v) in enumerate(rest)
            next_remaining = copy(rest)
            deleteat!(next_remaining, idx)
            push!(chosen, minmax(u, v))
            rec(next_remaining, chosen)
            pop!(chosen)
        end
    end
    rec(sort(vertices), Tuple{Int, Int}[])
    return out
end

function n8_t_surgery_rep(A_vec::Vector{Int}, matching::Vector{Tuple{Int, Int}}, masks::Vector{Int})
    nv = 8
    A = Set(A_vec)
    B_vec = [v for v in 1:nv if !(v in A)]
    edges = Set{Tuple{Int, Int}}()
    for x in A_vec, y in B_vec
        push!(edges, minmax(x, y))
    end
    for (r, (u, v)) in enumerate(matching)
        push!(edges, minmax(u, v))
        mask = masks[r]
        for (pos, x) in enumerate(A_vec)
            in_P = ((mask >> (pos - 1)) & 1) == 1
            delete!(edges, in_P ? minmax(x, u) : minmax(x, v))
        end
    end
    return graph_rep_from_edges(nv, sort!(collect(edges)))
end

function n8_t_surgery_selected_inputs(t::Int; include_base::Bool = true)
    nv = 8
    a = 4
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    selected = Set{BigInt}([big(1), complete_graph_rep(nv)])
    negatives = Set{BigInt}()
    rows = NamedTuple[]
    masks = collect(1:((1 << a) - 2))

    for A_vec in combinations_vec(collect(1:nv), a)
        A = Set(A_vec)
        B_vec = [v for v in 1:nv if !(v in A)]
        if include_base
            base = complete_bipartite_rep(nv, a)
            # Relabel the fixed construction by explicitly rebuilding from A.
            base_edges = Tuple{Int, Int}[]
            for x in A_vec, y in B_vec
                push!(base_edges, minmax(x, y))
            end
            base = BigInt(graph_rep_from_edges(nv, sort!(base_edges)))
            push!(selected, base)
            push!(negatives, base)
            for edge_idx in completing_edge_indices(nv, base, E, edge_index)
                push!(selected, BigInt(add_edge_rep(base, edge_idx)))
            end
        end

        for matching in matchings_of_size(B_vec, t)
            function rec_masks(chosen::Vector{Int}, depth::Int)
                if depth > t
                    rep = BigInt(n8_t_surgery_rep(A_vec, matching, chosen))
                    push!(selected, rep)
                    push!(negatives, rep)
                    c = triangle_completing_edges(nv, rep, E, edge_index)
                    expected = binomial(nv - 1, 2) - (a - 1) * (nv - a - 1) + t * (a - 1)
                    c == expected || error("bad c(G) for rep=$rep: got $c expected $expected")
                    for edge_idx in completing_edge_indices(nv, rep, E, edge_index)
                        push!(selected, BigInt(add_edge_rep(rep, edge_idx)))
                    end
                    push!(rows, (
                        rep = rep,
                        edges = count_ones(rep - 1),
                        completing = c,
                        expected = expected,
                    ))
                    return
                end
                for mask in masks
                    push!(chosen, mask)
                    rec_masks(chosen, depth + 1)
                    pop!(chosen)
                end
            end
            rec_masks(Int[], 1)
        end
    end
    return sort!(collect(selected)), sort!(collect(negatives)), rows
end

function close_under_vertex_generators(nv::Int, selected::Vector{BigInt})
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    gens = Vector{Int}[]
    for i in 1:(nv - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        push!(gens, sigma)
    end

    closed = Set{BigInt}(selected)
    queue = collect(selected)
    head = 1
    while head <= length(queue)
        x = queue[head]
        head += 1
        for sigma in gens
            y = BigInt(apply_vertex_perm_to_graph_index(sigma, x, E, edge_index))
            if !(y in closed)
                push!(closed, y)
                push!(queue, y)
            end
        end
    end
    return sort!(collect(closed))
end

function run_n8_t_surgery_direct(;
    t::Int = 1,
    include_base::Bool = true,
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    selected, negatives, rows = n8_t_surgery_selected_inputs(t; include_base = include_base)
    initial_selected = length(selected)
    selected = close_under_vertex_generators(8, selected)
    println("n=8 direct K_{4,4} t-surgery t=$t include_base=$include_base")
    println("  negative labeled inputs=$(length(negatives))")
    println("  selected labeled inputs=$initial_selected")
    println("  symmetry-closed labeled inputs=$(length(selected))")
    println("  surgery constructions before dedup=$(length(rows))")
    if !isempty(rows)
        cvals = sort(unique(r.completing for r in rows))
        evals = sort(unique(r.edges for r in rows))
        println("  surgery edge counts=$evals")
        println("  surgery c(G) values=$cvals expected=$(first(rows).expected)")
    end
    result = triangle_adversary_dual_restricted(
        8;
        solver = solver,
        verbose = verbose,
        explicit_selected_full = selected,
        time_limit_sec = time_limit_sec,
    )
    println(
        "N8_T_SURGERY_DIRECT t=$t include_base=$include_base value=$(result.value) " *
        "status=$(result.status) inputs=$(result.localN) pairs=$(result.n_pair_orbits) " *
        "maxblock=$(maximum(b[2] for b in result.blocks))",
    )
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    t = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
    include_base = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : true
    solver = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :mosek
    verbose = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : true
    time_limit_sec = length(ARGS) >= 5 ? parse_optional_time_limit(ARGS[5]) : nothing
    run_n8_t_surgery_direct(
        t = t,
        include_base = include_base,
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
end
