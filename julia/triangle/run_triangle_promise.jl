include(joinpath(@__DIR__, "run_triangle_symmetric.jl"))

function promise_orbit_reps(nv::Int)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => i for (i, e) in enumerate(E))
    fullN = 1 << length(E)
    orbit_sizes = graph_orbits_only_direct(nv, E, edge_index, fullN)

    reps = Set{Int}()
    negs = Int[]
    poss = Set{Int}()
    for rep in sort(collect(keys(orbit_sizes)))
        graph_has_triangle(nv, edge_index, rep) && continue
        completing = completing_edge_indices(nv, rep, E, edge_index)
        isempty(completing) && continue

        push!(reps, rep)
        push!(negs, rep)
        for e in completing
            push!(reps, canonical_graph_rep(nv, add_edge_rep(rep, e)))
        end
    end

    for rep in reps
        graph_has_triangle(nv, edge_index, rep) && push!(poss, rep)
    end
    return sort(collect(reps)), sort(negs), sort(collect(poss))
end

function run_triangle_promise(
    nv::Int;
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    reps, negs, poss = promise_orbit_reps(nv)
    println(
        "PROMISE n=$nv solver=$solver reps=$(length(reps)) " *
        "neg_orbits=$(length(negs)) pos_orbits=$(length(poss))",
    )
    verbose && println("negative reps=$negs")
    verbose && println("positive reps=$poss")

    result = solve_triangle_with_orbits(
        nv,
        reps;
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
    println(
        "RESULT promise n=$nv solver=$solver value=$(result.value) " *
        "status=$(result.status) inputs=$(result.localN) " *
        "pairs=$(result.n_pair_orbits) " *
        "maxblock=$(maximum(b[2] for b in result.blocks))",
    )
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    1 <= length(ARGS) <= 4 || error(
        "Usage: julia --project=. run_triangle_promise.jl <n> [solver=mosek] [verbose=true] [time_limit_sec=7200|none|0]",
    )
    nv = parse(Int, ARGS[1])
    solver = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :mosek
    verbose = length(ARGS) >= 3 ? parse(Bool, ARGS[3]) : true
    time_limit_sec = if length(ARGS) >= 4
        lowercase(ARGS[4]) in ("none", "nothing", "inf", "infinite", "0") ? nothing : parse(Float64, ARGS[4])
    else
        7200.0
    end
    run_triangle_promise(nv; solver = solver, verbose = verbose, time_limit_sec = time_limit_sec)
end
