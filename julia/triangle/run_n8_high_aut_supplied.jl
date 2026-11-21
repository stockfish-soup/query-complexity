include(joinpath(@__DIR__, "run_triangle_symmetric.jl"))

const N8_HIGH_AUT_ROWS = [
    (rep = 128, aut = 5040, orbit = 8, edges = 7, completing = 21, positives = BigInt[256]),
    (rep = 8191, aut = 1440, orbit = 28, edges = 12, completing = 16, positives = BigInt[8192, 16383]),
    (rep = 4185721, aut = 1152, orbit = 35, edges = 16, completing = 12, positives = BigInt[4185722]),
    (rep = 64, aut = 720, orbit = 56, edges = 6, completing = 15, positives = BigInt[192]),
    (rep = 262013, aut = 720, orbit = 56, edges = 15, completing = 13, positives = BigInt[262014, 524157]),
    (rep = 253, aut = 240, orbit = 168, edges = 6, completing = 10, positives = BigInt[8823]),
    (rep = 32, aut = 240, orbit = 168, edges = 5, completing = 10, positives = BigInt[160]),
    (rep = 4031, aut = 240, orbit = 168, edges = 10, completing = 11, positives = BigInt[4032, 12223]),
]

function run_n8_high_aut_supplied(;
    rounds::Int = length(N8_HIGH_AUT_ROWS),
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    nv = 8
    reps = BigInt[big(1), complete_graph_rep(nv)]
    results = NamedTuple[]
    println("n=8 high-automorphism supplied candidates")
    println("round rep aut orbit edges completing positives")
    for (rank, row) in enumerate(N8_HIGH_AUT_ROWS)
        println("$rank $(row.rep) $(row.aut) $(row.orbit) $(row.edges) $(row.completing) $(row.positives)")
    end

    for (round, row) in enumerate(Iterators.take(N8_HIGH_AUT_ROWS, rounds))
        append!(reps, BigInt[row.rep])
        append!(reps, row.positives)
        sort!(unique!(reps))
        println("\nRound $round: add rep=$(row.rep) aut=$(row.aut) c=$(row.completing)")
        println("  orbit reps=$(length(reps)): $reps")
        result = solve_triangle_with_orbits(
            nv,
            reps;
            solver = solver,
            verbose = verbose,
            time_limit_sec = time_limit_sec,
        )
        maxblock = maximum(b[2] for b in result.blocks)
        push!(results, (
            round = round,
            rep = row.rep,
            aut = row.aut,
            completing = row.completing,
            value = result.value,
            status = result.status,
            orbits = length(reps),
            inputs = result.localN,
            pairs = result.n_pair_orbits,
            maxblock = maxblock,
        ))
        println(
            "N8_HIGH_AUT round=$round rep=$(row.rep) aut=$(row.aut) c=$(row.completing) " *
            "value=$(result.value) status=$(result.status) orbits=$(length(reps)) " *
            "inputs=$(result.localN) pairs=$(result.n_pair_orbits) maxblock=$maxblock",
        )
        GC.gc()
    end

    println("\nSummary")
    println("round rep aut c value status orbits inputs pairs maxblock")
    for r in results
        println("$(r.round) $(r.rep) $(r.aut) $(r.completing) $(r.value) $(r.status) $(r.orbits) $(r.inputs) $(r.pairs) $(r.maxblock)")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    rounds = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : length(N8_HIGH_AUT_ROWS)
    solver = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :mosek
    verbose = length(ARGS) >= 3 ? parse(Bool, ARGS[3]) : true
    time_limit_sec = length(ARGS) >= 4 ? parse_optional_time_limit(ARGS[4]) : nothing
    run_n8_high_aut_supplied(
        rounds = rounds,
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
end
