include(joinpath(@__DIR__, "run_triangle_symmetric.jl"))

function complement_components(nv::Int, rep::Integer)
    present = Set(graph_edges(nv, rep))
    adjacency = [Int[] for _ in 1:nv]
    for i in 1:nv, j in (i + 1):nv
        edge = minmax(i, j)
        edge in present && continue
        push!(adjacency[i], j)
        push!(adjacency[j], i)
    end
    seen = falses(nv)
    comps = Int[]
    for v in 1:nv
        seen[v] && continue
        queue = [v]
        seen[v] = true
        head = 1
        count = 0
        while head <= length(queue)
            u = queue[head]
            head += 1
            count += 1
            for w in adjacency[u]
                if !seen[w]
                    seen[w] = true
                    push!(queue, w)
                end
            end
        end
        push!(comps, count)
    end
    return sort!(comps; rev = true)
end

function structural_label(nv::Int, rep::Integer, output::Int, completing::Int, degrees)
    edges = count_ones(rep - 1)
    tri = triangle_count_of_graph(nv, Dict(e => k for (k, e) in enumerate(edges_of_Kn(nv))), rep)
    if output == 0
        if completing > 0
            return "triangle-free boundary, c=$completing"
        elseif edges == 0
            return "empty graph"
        else
            return "triangle-free interior"
        end
    end
    if tri == 1
        return "single-triangle positive"
    elseif edges == binomial(nv, 2)
        return "complete graph"
    else
        return "multi-triangle positive"
    end
end

function analyze_n5_full_scores(; solver::Symbol = :mosek, verbose::Bool = false, time_limit_sec = nothing)
    nv = 5
    result = run_triangle_full(nv; solver = solver, verbose = verbose, time_limit_sec = time_limit_sec)
    diagnostics = sort(result.orbit_diagnostics; by = d -> d.importance, rev = true)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))

    println("\nDetailed n=5 full-orbit score table")
    println("rank rep f orbit aut edges tri c degrees complement label obj_mult cross_mass activity score edges_list")
    for (rank, d) in enumerate(diagnostics)
        edges = graph_edges(nv, d.full_rep)
        present = Set(edges)
        completing = d.output == 0 ? length(completing_edge_set(nv, present)) : 0
        label = structural_label(nv, d.full_rep, d.output, completing, d.degrees)
        comp = join(complement_components(nv, d.full_rep), ":")
        println(
            "$rank $(d.full_rep) $(d.output) $(d.orbit_size) $(d.automorphisms) " *
            "$(d.edges) $(d.triangles) $completing $(join(d.degrees, ':')) $comp " *
            "\"$label\" $(round(d.objective_multiplier, sigdigits=6)) " *
            "$(round(d.crossing_multiplier_mass, sigdigits=6)) " *
            "$(round(d.activity, sigdigits=6)) $(round(d.importance, sigdigits=6)) $edges",
        )
    end

    println("\nHighest-score structure summary")
    for (rank, d) in enumerate(Iterators.take(diagnostics, 10))
        edges = graph_edges(nv, d.full_rep)
        present = Set(edges)
        completing = d.output == 0 ? length(completing_edge_set(nv, present)) : 0
        label = structural_label(nv, d.full_rep, d.output, completing, d.degrees)
        println(
            "$rank. rep=$(d.full_rep), f=$(d.output), score=$(round(d.importance, sigdigits=5)), " *
            "edges=$(d.edges), triangles=$(d.triangles), c=$completing, " *
            "degrees=$(join(d.degrees, ':')), label=$label, edges=$edges",
        )
    end

    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    solver = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :mosek
    verbose = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : false
    time_limit_sec = length(ARGS) >= 3 ? parse_optional_time_limit(ARGS[3]) : nothing
    analyze_n5_full_scores(solver = solver, verbose = verbose, time_limit_sec = time_limit_sec)
end
