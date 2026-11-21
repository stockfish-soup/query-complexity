include(joinpath(@__DIR__, "run_balanced_max_t_surgery_batch.jl"))

using Printf

function append_positive_rows(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "family,n,value,status,orbits,inputs,pair_orbits,pair_members,maxblock")
        for r in rows
            println(io, join((
                r.family,
                r.n,
                r.value,
                r.status,
                r.orbits,
                r.inputs,
                r.pair_orbits,
                r.pair_members,
                r.maxblock,
            ), ","))
        end
    end
end

function solve_positive_for_reps(
    family::AbstractString,
    nv::Int,
    reps;
    solver::Symbol = :mosek,
    verbose::Bool = false,
    max_pair_members::Int = 2_000_000,
)
    m = binomial(nv, 2)
    reps_for_primal = m < Sys.WORD_SIZE ? Int.(reps) : reps
    result = triangle_adversary_primal_restricted(
        nv,
        reps_for_primal;
        solver = solver,
        verbose = verbose,
        max_pair_members = max_pair_members,
        positive_gamma = true,
    )
    return (
        family = family,
        n = nv,
        value = result.value,
        status = string(result.status),
        orbits = length(reps_for_primal),
        inputs = result.localN,
        pair_orbits = result.n_pair_orbits,
        pair_members = result.pair_members,
        maxblock = maximum(b[2] for b in result.blocks),
    )
end

function run_positive_adversary_benchmarks(;
    start_n::Int = 4,
    max_n::Int = 7,
    solver::Symbol = :mosek,
    verbose::Bool = false,
    max_pair_members::Int = 2_000_000,
    out_csv::AbstractString = joinpath(@__DIR__, "results", "positive_adversary_benchmarks.csv"),
)
    rows = NamedTuple[]
    for nv in start_n:max_n
        println("\nPositive adversary benchmark n=$nv")

        try
            surgery_reps, _, _ = balanced_max_t_surgery_orbit_reps(nv; include_base = true, verbose = false)
            row = solve_positive_for_reps(
                "balanced max-t surgery",
                nv,
                surgery_reps;
                solver = solver,
                verbose = verbose,
                max_pair_members = max_pair_members,
            )
            push!(rows, row)
            println("  surgery positive ADV=$(row.value) inputs=$(row.inputs) pair_members=$(row.pair_members)")
        catch err
            println("  surgery positive ADV failed for n=$nv: $err")
        end

        try
            cb_reps = complete_bipartite_all_k_reps(nv, fld(nv, 2))
            row = solve_positive_for_reps(
                "complete bipartite",
                nv,
                cb_reps;
                solver = solver,
                verbose = verbose,
                max_pair_members = max_pair_members,
            )
            push!(rows, row)
            println("  complete bipartite positive ADV=$(row.value) inputs=$(row.inputs) pair_members=$(row.pair_members)")
        catch err
            println("  complete bipartite positive ADV failed for n=$nv: $err")
        end

        append_positive_rows(out_csv, rows)
        GC.gc()
    end
    println("\nPositive adversary results")
    println("family n value status orbits inputs pair_orbits pair_members maxblock")
    for r in rows
        println("$(r.family) $(r.n) $(r.value) $(r.status) $(r.orbits) $(r.inputs) $(r.pair_orbits) $(r.pair_members) $(r.maxblock)")
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    start_n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
    max_n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7
    solver = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :mosek
    verbose = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : false
    max_pair_members = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2_000_000
    run_positive_adversary_benchmarks(
        start_n = start_n,
        max_n = max_n,
        solver = solver,
        verbose = verbose,
        max_pair_members = max_pair_members,
    )
end
