include(joinpath(@__DIR__, "run_balanced_max_t_surgery_batch.jl"))

using Printf

function solve_general_for_reps(nv::Int, reps; solver::Symbol = :mosek, verbose::Bool = false)
    result = solve_triangle_with_orbits(
        nv,
        reps;
        solver = solver,
        verbose = verbose,
        time_limit_sec = nothing,
    )
    return (
        value = result.value,
        status = string(result.status),
        inputs = result.localN,
        pair_orbits = result.n_pair_orbits,
        maxblock = maximum(b[2] for b in result.blocks),
    )
end

function solve_positive_for_reps(nv::Int, reps; solver::Symbol = :mosek, verbose::Bool = false, max_pair_members::Int = 10_000_000)
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
        value = result.value,
        status = string(result.status),
        inputs = result.localN,
        pair_orbits = result.n_pair_orbits,
        pair_members = result.pair_members,
        maxblock = maximum(b[2] for b in result.blocks),
    )
end

function no_base_max_t_surgery_reps(nv::Int)
    reps, rows, _ = balanced_max_t_surgery_orbit_reps(nv; include_base = false, verbose = true)
    isempty(rows) && error("no max-t surgery rows generated for n=$nv")
    max_c = maximum(row.completing for row in rows)
    t = maximum(row.t for row in rows)
    return reps, rows, max_c, t
end

function run_requested_positive_ratios(;
    solver::Symbol = :mosek,
    verbose::Bool = false,
    max_pair_members::Int = 10_000_000,
    out_csv::AbstractString = joinpath(@__DIR__, "results", "requested_positive_ratios.csv"),
)
    results = NamedTuple[]
    for nv in (6, 7)
        println("\n=== n=$nv complete bipartite cumulative ===")
        cb_reps = complete_bipartite_all_k_reps(nv, fld(nv, 2))
        gen = solve_general_for_reps(nv, cb_reps; solver = solver, verbose = verbose)
        pos = solve_positive_for_reps(nv, cb_reps; solver = solver, verbose = verbose, max_pair_members = max_pair_members)
        push!(results, (
            family = "complete bipartite",
            n = nv,
            t = 0,
            max_c = 0,
            general = gen.value,
            positive = pos.value,
            ratio = gen.value / pos.value,
            general_status = gen.status,
            positive_status = pos.status,
            orbits = length(cb_reps),
            inputs = gen.inputs,
            pair_orbits = gen.pair_orbits,
            pair_members = pos.pair_members,
            maxblock = gen.maxblock,
        ))
        println("RESULT_RATIO family=complete-bipartite n=$nv general=$(gen.value) positive=$(pos.value) ratio=$(gen.value / pos.value)")

        println("\n=== n=$nv balanced max-t surgery only, no K_ab base ===")
        surg_reps, _, max_c, t = no_base_max_t_surgery_reps(nv)
        gen = solve_general_for_reps(nv, surg_reps; solver = solver, verbose = verbose)
        pos = solve_positive_for_reps(nv, surg_reps; solver = solver, verbose = verbose, max_pair_members = max_pair_members)
        push!(results, (
            family = "max-t surgery no base",
            n = nv,
            t = t,
            max_c = max_c,
            general = gen.value,
            positive = pos.value,
            ratio = gen.value / pos.value,
            general_status = gen.status,
            positive_status = pos.status,
            orbits = length(surg_reps),
            inputs = gen.inputs,
            pair_orbits = gen.pair_orbits,
            pair_members = pos.pair_members,
            maxblock = gen.maxblock,
        ))
        println("RESULT_RATIO family=max-t-surgery-no-base n=$nv t=$t max_c=$max_c general=$(gen.value) positive=$(pos.value) ratio=$(gen.value / pos.value)")

        open(out_csv, "w") do io
            println(io, "family,n,t,max_c,general,positive,ratio,general_status,positive_status,orbits,inputs,pair_orbits,pair_members,maxblock")
            for r in results
                println(io, join((
                    r.family, r.n, r.t, r.max_c, r.general, r.positive, r.ratio,
                    r.general_status, r.positive_status, r.orbits, r.inputs,
                    r.pair_orbits, r.pair_members, r.maxblock,
                ), ","))
            end
        end
        GC.gc()
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    solver = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :mosek
    verbose = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : false
    max_pair_members = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10_000_000
    run_requested_positive_ratios(
        solver = solver,
        verbose = verbose,
        max_pair_members = max_pair_members,
    )
end
