include(joinpath(@__DIR__, "run_triangle_symmetric.jl"))
include(joinpath(@__DIR__, "run_balanced_complete_batch.jl"))

using Dates
using Printf

function mask_tuples(a::Int, t::Int)
    t == 0 && return [Int[]]
    masks = collect(1:((1 << a) - 2))
    out = Vector{Vector{Int}}()
    current = Int[]
    function rec(depth::Int)
        if depth > t
            push!(out, copy(current))
            return
        end
        for mask in masks
            push!(current, mask)
            rec(depth + 1)
            pop!(current)
        end
    end
    rec(1)
    return out
end

function balanced_max_t_surgery_orbit_reps(nv::Int; include_base::Bool = true, verbose::Bool = true)
    a = fld(nv, 2)
    b = nv - a
    t = fld(b, 2)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    reps = BigInt[big(1), complete_graph_rep(nv)]
    negative_rows = NamedTuple[]

    if include_base
        base = canonical_graph_rep(nv, complete_bipartite_rep(nv, a))
        base_pos = BigInt[
            canonical_graph_rep(nv, add_edge_rep(base, edge_idx))
            for edge_idx in completing_edge_indices(nv, base, E, edge_index)
        ]
        push!(reps, base)
        append!(reps, base_pos)
        push!(negative_rows, (
            kind = "K_{$a,$b}",
            rep = BigInt(base),
            t = 0,
            edges = count_ones(base - 1),
            completing = triangle_completing_edges(nv, base, E, edge_index),
            positives = sort!(unique!(base_pos)),
        ))
    end

    surgery_rows = NamedTuple[]
    if t >= 1 && a >= 2
        expected = binomial(nv - 1, 2) - (a - 1) * (b - 1) + t * (a - 1)
        for masks in mask_tuples(a, t)
            rep = canonical_graph_rep(nv, t_surgery_graph_rep(nv, a, t, masks))
            c = triangle_completing_edges(nv, rep, E, edge_index)
            push!(surgery_rows, (
                rep = BigInt(rep),
                masks = copy(masks),
                edges = count_ones(rep - 1),
                completing = c,
                expected = expected,
            ))
        end
        max_c = maximum(row.completing for row in surgery_rows)
        surgery_reps = sort!(unique(row.rep for row in surgery_rows if row.completing == max_c))
        for rep in surgery_reps
            pos = BigInt[
                canonical_graph_rep(nv, add_edge_rep(rep, edge_idx))
                for edge_idx in completing_edge_indices(nv, rep, E, edge_index)
            ]
            push!(reps, rep)
            append!(reps, pos)
            push!(negative_rows, (
                kind = "$(t)-surgery(K_{$a,$b})",
                rep = rep,
                t = t,
                edges = count_ones(rep - 1),
                completing = triangle_completing_edges(nv, rep, E, edge_index),
                positives = sort!(unique!(pos)),
            ))
        end
        verbose && println(
            "  surgery masks=$(length(surgery_rows)) max_c=$max_c expected_formula=$expected " *
            "kept_surgery_orbits=$(length(surgery_reps))",
        )
    end

    reps = sort!(unique!(reps))
    if verbose
        println("balanced max-t surgery n=$nv a=$a b=$b t=$t include_base=$include_base")
        println("  reps=$(length(reps)): $reps")
        println("  negative constructions:")
        for row in negative_rows
            println(
                "    $(row.kind) rep=$(row.rep) t=$(row.t) edges=$(row.edges) " *
                "c=$(row.completing) positives=$(row.positives)",
            )
        end
    end
    return reps, negative_rows, surgery_rows
end

function run_balanced_max_t_surgery_batch(;
    start_n::Int = 4,
    max_n::Int = 8,
    include_base::Bool = true,
    solver::Symbol = :mosek,
    verbose::Bool = false,
    time_limit_sec = nothing,
    out_csv::AbstractString = joinpath(@__DIR__, "results", "balanced_max_t_surgery_values.csv"),
    out_svg::AbstractString = joinpath(@__DIR__, "results", "balanced_max_t_surgery_values.svg"),
)
    rows = NamedTuple[]
    println("Balanced max-t surgery batch n=$start_n:$max_n include_base=$include_base")
    for nv in start_n:max_n
        t0 = time()
        try
            reps, negative_rows, _ = balanced_max_t_surgery_orbit_reps(nv; include_base = include_base, verbose = true)
            result = solve_triangle_with_orbits(
                nv,
                reps;
                solver = solver,
                verbose = verbose,
                time_limit_sec = time_limit_sec,
            )
            a = fld(nv, 2)
            b = nv - a
            t = fld(b, 2)
            max_c = maximum(row.completing for row in negative_rows)
            row = (
                n = nv,
                a = a,
                b = b,
                t = t,
                max_c = max_c,
                value = result.value,
                status = string(result.status),
                orbits = length(reps),
                inputs = result.localN,
                pair_orbits = result.n_pair_orbits,
                maxblock = maximum(block[2] for block in result.blocks),
                elapsed_sec = time() - t0,
            )
            push!(rows, row)
            println(
                "BALANCED_MAX_T_SURGERY n=$nv a=$a b=$b t=$t max_c=$max_c " *
                "value=$(row.value) status=$(row.status) orbits=$(row.orbits) " *
                "inputs=$(row.inputs) pairs=$(row.pair_orbits) maxblock=$(row.maxblock) " *
                "elapsed=$(round(row.elapsed_sec, digits=2))",
            )
            open(out_csv, "w") do io
                println(io, "n,a,b,t,max_c,value,status,orbits,inputs,pair_orbits,maxblock,elapsed_sec")
                for r in rows
                    println(io, join((
                        r.n, r.a, r.b, r.t, r.max_c, r.value, r.status,
                        r.orbits, r.inputs, r.pair_orbits, r.maxblock,
                        @sprintf("%.3f", r.elapsed_sec),
                    ), ","))
                end
            end
            write_simple_svg(out_svg, rows)
        catch err
            println("FAILED n=$nv after $(round(time() - t0, digits=2))s: $err")
            break
        finally
            GC.gc()
        end
    end

    println("\nSummary")
    println("n a b t max_c value status orbits inputs pairs maxblock elapsed_sec")
    for r in rows
        println("$(r.n) $(r.a) $(r.b) $(r.t) $(r.max_c) $(r.value) $(r.status) $(r.orbits) $(r.inputs) $(r.pair_orbits) $(r.maxblock) $(round(r.elapsed_sec, digits=3))")
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    start_n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
    max_n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
    include_base = length(ARGS) >= 3 ? parse(Bool, ARGS[3]) : true
    solver = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :mosek
    verbose = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : false
    time_limit_sec = length(ARGS) >= 6 ? parse_optional_time_limit(ARGS[6]) : nothing
    run_balanced_max_t_surgery_batch(
        start_n = start_n,
        max_n = max_n,
        include_base = include_base,
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
end
