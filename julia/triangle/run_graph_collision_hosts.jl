include(joinpath(@__DIR__, "graph_collision_adversary_dual.jl"))

using Random
using TOML

function _adjacent_swap(nv::Int, i::Int)
    sigma = collect(1:nv)
    sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
    return sigma
end

function _complete_bipartite_host(a::Int, b::Int)
    nv = a + b
    host_edges = [(u, v) for u in 1:a for v in (a + 1):nv]
    generators = Vector{Int}[]
    for i in 1:(a - 1)
        push!(generators, _adjacent_swap(nv, i))
    end
    for i in (a + 1):(nv - 1)
        push!(generators, _adjacent_swap(nv, i))
    end
    if a == b
        push!(generators, [i <= a ? i + a : i - a for i in 1:nv])
    end
    isempty(generators) && push!(generators, collect(1:nv))
    return host_edges, generators
end

function _matching_host(m::Int)
    nv = 2m
    host_edges = [(i, m + i) for i in 1:m]
    generators = Vector{Int}[]
    for i in 1:(m - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        sigma[m + i], sigma[m + i + 1] = sigma[m + i + 1], sigma[m + i]
        push!(generators, sigma)
    end
    for i in 1:m
        sigma = collect(1:nv)
        sigma[i], sigma[m + i] = sigma[m + i], sigma[i]
        push!(generators, sigma)
    end
    isempty(generators) && push!(generators, collect(1:nv))
    return host_edges, generators
end

function _crown_host(m::Int)
    nv = 2m
    host_edges = [(i, m + j) for i in 1:m for j in 1:m if i != j]
    generators = Vector{Int}[]
    for i in 1:(m - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        sigma[m + i], sigma[m + i + 1] = sigma[m + i + 1], sigma[m + i]
        push!(generators, sigma)
    end
    push!(generators, [i <= m ? i + m : i - m for i in 1:nv])
    return host_edges, generators
end

function _path_host(nv::Int)
    host_edges = [(i, i + 1) for i in 1:(nv - 1)]
    generators = [[nv + 1 - i for i in 1:nv]]
    return host_edges, generators
end

function _cycle_host(nv::Int)
    host_edges = [(i, i == nv ? 1 : i + 1) for i in 1:nv]
    generators = [
        [i == nv ? 1 : i + 1 for i in 1:nv],
        [mod1(2 - i, nv) for i in 1:nv],
    ]
    return host_edges, generators
end

function _host_summary(nv::Int, host_edges, generators)
    edges = _normalize_host_edges(nv, host_edges)
    vertex_elements = _gc_group_elements(nv, generators)
    degrees = zeros(Int, nv)
    for (u, v) in edges
        degrees[u] += 1
        degrees[v] += 1
    end
    return (
        edges = length(edges),
        degrees = sort!(degrees; rev = true),
        automorphisms = length(vertex_elements),
        query_orbits = _query_orbits(nv, vertex_elements),
    )
end

function solve_graph_collision_host(
    name::AbstractString,
    nv::Int,
    host_edges,
    generators;
    solver::Symbol = :mosek,
    verbose::Bool = false,
    time_limit_sec = nothing,
)
    summary = _host_summary(nv, host_edges, generators)
    println(
        "HOST $name nv=$nv edges=$(summary.edges) degrees=$(summary.degrees) " *
        "aut=$(summary.automorphisms) query_orbits=$(summary.query_orbits)",
    )
    result = graph_collision_adversary_dual(
        nv,
        host_edges,
        generators;
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
    println(
        "RESULT $name nv=$nv value=$(result.value) status=$(result.status) " *
        "inputs=$(result.input_count) input_orbits=$(result.n_input_orbits) " *
        "pair_orbits=$(result.n_pair_orbits) blocks=$(result.blocks)",
    )
    return result
end

function run_bipartite_symmetric_hosts(
    nv::Int;
    solver::Symbol = :mosek,
    verbose::Bool = false,
    time_limit_sec = nothing,
)
    hosts = Tuple{String, Vector{Tuple{Int, Int}}, Vector{Vector{Int}}}[]

    for a in 1:fld(nv, 2)
        b = nv - a
        edges, generators = _complete_bipartite_host(a, b)
        push!(hosts, ("K_$a,$b", edges, generators))
    end
    if iseven(nv)
        m = nv ÷ 2
        edges, generators = _matching_host(m)
        push!(hosts, ("matching_$m", edges, generators))
        m >= 2 && begin
            edges, generators = _crown_host(m)
            push!(hosts, ("crown_$m", edges, generators))
        end
    end
    edges, generators = _path_host(nv)
    push!(hosts, ("path_$nv", edges, generators))
    iseven(nv) && begin
        edges, generators = _cycle_host(nv)
        push!(hosts, ("cycle_$nv", edges, generators))
    end

    results = NamedTuple[]
    for (name, edges, generators) in hosts
        try
            solve = solve_graph_collision_host(
                name,
                nv,
                edges,
                generators;
                solver = solver,
                verbose = verbose,
                time_limit_sec = time_limit_sec,
            )
            push!(results, (
                name = name,
                value = Float64(solve.value),
                status = string(solve.status),
                input_orbits = solve.n_input_orbits,
                pair_orbits = solve.n_pair_orbits,
                blocks = solve.blocks,
            ))
        catch err
            println("FAILED $name nv=$nv error=$(typeof(err)): $err")
        end
        GC.gc()
    end
    sort!(results; by = row -> -row.value)
    println("\nTop symmetric bipartite hosts for nv=$nv:")
    for (rank, row) in enumerate(results)
        println(
            "  $rank. $(row.name) value=$(row.value) status=$(row.status) " *
            "input_orbits=$(row.input_orbits) pair_orbits=$(row.pair_orbits)",
        )
    end
    return results
end

function _all_vertex_permutations(nv::Int)
    generators = Perm[]
    for i in 1:(nv - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        push!(generators, _gc_permutation(sigma))
    end
    push!(generators, _gc_permutation(collect(1:nv)))
    group = collect(PermGroup(generators))
    return [[Int(vertex^g) for vertex in 1:nv] for g in group]
end

function _host_edges_of_Kn(nv::Int)
    return [(u, v) for u in 1:nv for v in (u + 1):nv]
end

function _apply_vertex_perm_to_host_bits(bits::Int, sigma, edges, edge_index)
    mapped = 0
    for (index, (u, v)) in enumerate(edges)
        ((bits >> (index - 1)) & 1) == 1 || continue
        a, b = sigma[u], sigma[v]
        a > b && ((a, b) = (b, a))
        mapped |= 1 << (edge_index[(a, b)] - 1)
    end
    return mapped
end

function enumerate_unlabeled_host_graphs(nv::Int)
    edges = _host_edges_of_Kn(nv)
    edge_index = Dict(edge => index for (index, edge) in enumerate(edges))
    permutations = _all_vertex_permutations(nv)
    graph_count = 1 << length(edges)
    seen = falses(graph_count)
    representatives = Int[]
    for bits in 0:(graph_count - 1)
        seen[bits + 1] && continue
        orbit = unique(
            _apply_vertex_perm_to_host_bits(bits, sigma, edges, edge_index)
            for sigma in permutations
        )
        seen[orbit .+ 1] .= true
        push!(representatives, minimum(orbit))
    end
    sort!(representatives)
    return representatives, edges, edge_index, permutations
end

function _host_edges_from_bits(bits::Int, edges)
    return [edge for (index, edge) in enumerate(edges) if ((bits >> (index - 1)) & 1) == 1]
end

function _host_automorphisms(bits::Int, edges, edge_index, permutations)
    return [
        sigma for sigma in permutations
        if _apply_vertex_perm_to_host_bits(bits, sigma, edges, edge_index) == bits
    ]
end

function _host_metadata(nv::Int, bits::Int, host_edges, automorphism_count::Int)
    degrees = zeros(Int, nv)
    adjacency = [Int[] for _ in 1:nv]
    for (u, v) in host_edges
        degrees[u] += 1
        degrees[v] += 1
        push!(adjacency[u], v)
        push!(adjacency[v], u)
    end
    seen = falses(nv)
    components = 0
    for start in 1:nv
        seen[start] && continue
        components += 1
        stack = [start]
        seen[start] = true
        while !isempty(stack)
            vertex = pop!(stack)
            for neighbor in adjacency[vertex]
                seen[neighbor] && continue
                seen[neighbor] = true
                push!(stack, neighbor)
            end
        end
    end
    return (
        representative = bits,
        edges = length(host_edges),
        edge_list = host_edges,
        degrees = sort!(degrees; rev = true),
        connected = components == 1,
        components = components,
        automorphisms = automorphism_count,
        orbit_size = factorial(nv) ÷ automorphism_count,
    )
end

function run_graph_collision_host_search(
    nv::Int;
    solver::Symbol = :mosek,
    sample_count::Union{Nothing, Int} = nothing,
    seed::Int = 1,
    output_path::AbstractString = joinpath(@__DIR__, "results", "graph_collision_host_results.toml"),
)
    representatives, all_edges, edge_index, permutations =
        enumerate_unlabeled_host_graphs(nv)
    nonempty = filter(!iszero, representatives)
    selected = if sample_count === nothing || sample_count >= length(nonempty)
        nonempty
    else
        rng = MersenneTwister(seed)
        sort!(shuffle(rng, nonempty)[1:sample_count])
    end

    mode = length(selected) == length(nonempty) ? "exhaustive" : "random"
    println("Graph-collision host search: nv=$nv, mode=$mode, solver=$solver")
    println("  solving $(length(selected)) / $(length(nonempty)) nonempty unlabeled hosts")

    results = NamedTuple[]
    for (run, bits) in enumerate(selected)
        host_edges = _host_edges_from_bits(bits, all_edges)
        automorphisms = _host_automorphisms(bits, all_edges, edge_index, permutations)
        metadata = _host_metadata(nv, bits, host_edges, length(automorphisms))
        print("  [$run/$(length(selected))] rep=$bits edges=$(metadata.edges) " *
              "degrees=$(metadata.degrees) aut=$(metadata.automorphisms) ... ")
        flush(stdout)
        solve = graph_collision_adversary_dual(
            nv,
            host_edges,
            automorphisms;
            solver = solver,
            verbose = false,
        )
        result = merge(metadata, (
            adversary_value = Float64(solve.value),
            status = string(solve.status),
            query_orbits = solve.query_orbits,
            input_orbits = solve.n_input_orbits,
            pair_orbits = solve.n_pair_orbits,
            blocks = solve.blocks,
        ))
        push!(results, result)
        println("$(result.adversary_value) [$(result.status)]")
        solve = nothing
        GC.gc()
    end

    sort!(results; by = result -> (-result.adversary_value, result.representative))
    serializable = Dict(
        "nv" => nv,
        "mode" => mode,
        "solver" => string(solver),
        "seed" => seed,
        "unlabeled_nonempty_hosts" => length(nonempty),
        "solved_hosts" => length(selected),
        "results" => [Dict(
            "rank" => rank,
            "representative" => result.representative,
            "adversary_value" => result.adversary_value,
            "status" => result.status,
            "edges" => result.edges,
            "edge_list" => [[u, v] for (u, v) in result.edge_list],
            "degrees" => result.degrees,
            "connected" => result.connected,
            "components" => result.components,
            "automorphisms" => result.automorphisms,
            "orbit_size" => result.orbit_size,
            "query_orbits" => result.query_orbits,
            "input_orbits" => result.input_orbits,
            "pair_orbits" => result.pair_orbits,
            "blocks" => [[[index, size] for (index, size) in family] for family in result.blocks],
        ) for (rank, result) in enumerate(results)],
    )
    open(output_path, "w") do io
        TOML.print(io, serializable; sorted = true)
    end

    println("\nTop host graphs:")
    for (rank, result) in enumerate(results[1:min(10, length(results))])
        println("  $rank. value=$(result.adversary_value) rep=$(result.representative) " *
                "edges=$(result.edges) degrees=$(result.degrees) host=$(result.edge_list)")
    end
    println("Results written to $output_path")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    !isempty(ARGS) || error(
        "Usage:\n" *
        "  julia --project=. run_graph_collision_hosts.jl all <nv> [solver=mosek] [output]\n" *
        "  julia --project=. run_graph_collision_hosts.jl random <nv> <count> [seed=1] [solver=mosek] [output]\n" *
        "  julia --project=. run_graph_collision_hosts.jl bipartite-symmetric <nv> [solver=mosek] [time_limit_sec=none]\n" *
        "  julia --project=. run_graph_collision_hosts.jl host <name> <nv> [solver=mosek] [time_limit_sec=none]"
    )
    mode = ARGS[1]
    if mode == "all"
        2 <= length(ARGS) <= 4 || error("Usage: ... all <nv> [solver=mosek] [output]")
        nv = parse(Int, ARGS[2])
        solver = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :mosek
        output = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "results", "graph_collision_host_results.toml")
        run_graph_collision_host_search(nv; solver = solver, output_path = output)
    elseif mode == "random"
        3 <= length(ARGS) <= 6 || error(
            "Usage: ... random <nv> <count> [seed=1] [solver=mosek] [output]"
        )
        nv = parse(Int, ARGS[2])
        count = parse(Int, ARGS[3])
        seed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
        solver = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :mosek
        output = length(ARGS) >= 6 ? ARGS[6] : joinpath(@__DIR__, "results", "graph_collision_host_results.toml")
        run_graph_collision_host_search(
            nv;
            solver = solver,
            sample_count = count,
            seed = seed,
            output_path = output,
        )
    elseif mode == "bipartite-symmetric"
        2 <= length(ARGS) <= 4 || error(
            "Usage: ... bipartite-symmetric <nv> [solver=mosek] [time_limit_sec=none]"
        )
        nv = parse(Int, ARGS[2])
        solver = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :mosek
        time_limit = length(ARGS) >= 4 && ARGS[4] != "none" ? parse(Float64, ARGS[4]) : nothing
        run_bipartite_symmetric_hosts(nv; solver = solver, time_limit_sec = time_limit)
    elseif mode == "host"
        3 <= length(ARGS) <= 5 || error(
            "Usage: ... host <name> <nv> [solver=mosek] [time_limit_sec=none]\n" *
            "Names: complete-a-b, matching, crown, path, cycle"
        )
        name = ARGS[2]
        nv = parse(Int, ARGS[3])
        solver = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :mosek
        time_limit = length(ARGS) >= 5 && ARGS[5] != "none" ? parse(Float64, ARGS[5]) : nothing
        if startswith(name, "complete-")
            parts = split(name, "-")
            length(parts) == 3 || error("complete host name must be complete-a-b")
            a = parse(Int, parts[2])
            b = parse(Int, parts[3])
            a + b == nv || error("complete-a-b has a+b != nv")
            edges, generators = _complete_bipartite_host(a, b)
        elseif name == "matching"
            iseven(nv) || error("matching host needs even nv")
            edges, generators = _matching_host(nv ÷ 2)
        elseif name == "crown"
            iseven(nv) || error("crown host needs even nv")
            edges, generators = _crown_host(nv ÷ 2)
        elseif name == "path"
            edges, generators = _path_host(nv)
        elseif name == "cycle"
            edges, generators = _cycle_host(nv)
        else
            error("Unknown host '$name'")
        end
        solve_graph_collision_host(
            name,
            nv,
            edges,
            generators;
            solver = solver,
            verbose = true,
            time_limit_sec = time_limit,
        )
    else
        error("Unknown mode '$mode'.")
    end
end
