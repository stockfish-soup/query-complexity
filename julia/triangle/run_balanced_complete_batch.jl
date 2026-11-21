include(joinpath(@__DIR__, "run_triangle_symmetric.jl"))

using Dates
using Printf

function parse_batch_time_limit(arg::AbstractString)
    return lowercase(arg) in ("none", "nothing", "inf", "infinite", "0") ? nothing : parse(Float64, arg)
end

function balanced_complete_reps(nv::Int)
    return complete_bipartite_round_reps(nv, fld(nv, 2))
end

function combinations_of_size(n::Int, k::Int)
    out = Vector{Vector{Int}}()
    current = Int[]
    function rec(start::Int, remaining::Int)
        if remaining == 0
            push!(out, copy(current))
            return
        end
        last_start = n - remaining + 1
        for v in start:last_start
            push!(current, v)
            rec(v + 1, remaining - 1)
            pop!(current)
        end
    end
    rec(1, k)
    return out
end

function balanced_complete_selected_inputs(nv::Int; include_anchors::Bool = true)
    a = fld(nv, 2)
    b = nv - a
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    onebit = big(1)
    selected = Set{BigInt}()

    include_anchors && push!(selected, big(1))
    include_anchors && push!(selected, complete_graph_rep(nv))

    for A_vec in combinations_of_size(nv, a)
        # If a == b, A and its complement define the same graph. We still
        # generate both sides here and let the Set deduplicate, because the
        # direct domain must remain closed under the vertex permutations used
        # by the symmetry reduction.
        A = Set(A_vec)
        B_vec = [v for v in 1:nv if !(v in A)]

        bits = big(0)
        for u in A_vec, v in B_vec
            x, y = minmax(u, v)
            bits |= onebit << (edge_index[(x, y)] - 1)
        end
        neg = bits + 1
        push!(selected, neg)

        for side in (A_vec, B_vec)
            length(side) < 2 && continue
            for i in 1:(length(side) - 1), j in (i + 1):length(side)
                u, v = minmax(side[i], side[j])
                pos = (bits | (onebit << (edge_index[(u, v)] - 1))) + 1
                push!(selected, pos)
            end
        end
    end
    return sort!(collect(selected))
end

function solve_balanced_complete_fast(
    nv::Int;
    solver::Symbol = :mosek,
    verbose::Bool = false,
    time_limit_sec = 7200.0,
    include_anchors::Bool = true,
)
    selected = balanced_complete_selected_inputs(nv; include_anchors = include_anchors)
    return triangle_adversary_dual_restricted(
        nv;
        solver = solver,
        verbose = verbose,
        explicit_selected_full = selected,
        time_limit_sec = time_limit_sec,
    )
end

function append_csv_row(path::AbstractString, row)
    new_file = !isfile(path)
    open(path, "a") do io
        if new_file
            println(io, "timestamp,n,value,status,orbits,inputs,pair_orbits,maxblock,elapsed_sec")
        end
        println(
            io,
            join((
                Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
                row.n,
                row.value,
                row.status,
                row.orbits,
                row.inputs,
                row.pair_orbits,
                row.maxblock,
                @sprintf("%.3f", row.elapsed_sec),
            ), ","),
        )
    end
end

function read_completed_rows(path::AbstractString)
    isfile(path) || return NamedTuple[]
    rows = NamedTuple[]
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue
        isempty(strip(line)) && continue
        cols = split(line, ",")
        length(cols) < 9 && continue
        push!(rows, (
            n = parse(Int, cols[2]),
            value = parse(Float64, cols[3]),
            status = cols[4],
            orbits = parse(Int, cols[5]),
            inputs = parse(Int, cols[6]),
            pair_orbits = parse(Int, cols[7]),
            maxblock = parse(Int, cols[8]),
            elapsed_sec = parse(Float64, cols[9]),
        ))
    end
    return rows
end

function write_simple_svg(path::AbstractString, rows)
    rows = sort(collect(rows); by = r -> r.n)
    isempty(rows) && return nothing

    width = 760
    height = 480
    margin_left = 70
    margin_right = 24
    margin_top = 24
    margin_bottom = 58
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    ns = [r.n for r in rows]
    vals = [r.value for r in rows]
    min_n, max_n = minimum(ns), maximum(ns)
    min_v = 0.0
    max_v = maximum(vals) * 1.08
    max_v == 0 && (max_v = 1.0)

    xcoord(n) = margin_left + (max_n == min_n ? 0.5 : (n - min_n) / (max_n - min_n)) * plot_w
    ycoord(v) = margin_top + (1 - (v - min_v) / (max_v - min_v)) * plot_h

    points = join(["$(round(xcoord(r.n), digits=2)),$(round(ycoord(r.value), digits=2))" for r in rows], " ")

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$(width/2)" y="18" text-anchor="middle" font-family="sans-serif" font-size="16">Balanced complete bipartite promise</text>""")
        println(io, """<line x1="$margin_left" y1="$(height-margin_bottom)" x2="$(width-margin_right)" y2="$(height-margin_bottom)" stroke="#111827"/>""")
        println(io, """<line x1="$margin_left" y1="$margin_top" x2="$margin_left" y2="$(height-margin_bottom)" stroke="#111827"/>""")
        for r in rows
            x = xcoord(r.n)
            y = ycoord(r.value)
            println(io, """<circle cx="$x" cy="$y" r="4.5" fill="#2563eb"/>""")
            println(io, """<text x="$x" y="$(y-10)" text-anchor="middle" font-family="sans-serif" font-size="11">$(round(r.value, digits=3))</text>""")
            println(io, """<text x="$x" y="$(height-margin_bottom+18)" text-anchor="middle" font-family="sans-serif" font-size="11">$(r.n)</text>""")
        end
        length(rows) >= 2 && println(io, """<polyline points="$points" fill="none" stroke="#2563eb" stroke-width="2"/>""")
        println(io, """<text x="$(width/2)" y="$(height-16)" text-anchor="middle" font-family="sans-serif" font-size="13">n</text>""")
        println(io, """<text x="18" y="$(height/2)" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 18 $(height/2))">ADV value</text>""")
        println(io, "</svg>")
    end
    return path
end

function run_balanced_complete_batch(;
    start_n::Int = 4,
    max_n::Int = 20,
    step::Int = 1,
    solver::Symbol = :mosek,
    verbose::Bool = false,
    time_limit_sec = 7200.0,
    out_csv::AbstractString = joinpath(@__DIR__, "results", "balanced_complete_bipartite_values.csv"),
    out_svg::AbstractString = joinpath(@__DIR__, "results", "balanced_complete_bipartite_values.svg"),
    resume::Bool = true,
)
    step > 0 || error("step must be positive")
    completed = resume ? Set(r.n for r in read_completed_rows(out_csv)) : Set{Int}()
    rows = read_completed_rows(out_csv)

    println("Balanced complete bipartite batch")
    println("  n range: $start_n:$step:$max_n")
    println("  solver=$solver time_limit_sec=$(time_limit_sec === nothing ? "none" : time_limit_sec)")
    println("  csv=$out_csv")
    println("  svg=$out_svg")

    for nv in start_n:step:max_n
        if nv in completed
            println("\nSkipping n=$nv; already present in $out_csv")
            continue
        end

        println("\n=== n=$nv, K_{$(fld(nv,2)),$(nv - fld(nv,2))} + completing edges ===")
        selected = balanced_complete_selected_inputs(nv)
        println("explicit inputs=$(length(selected))")
        t0 = time()
        try
            result = triangle_adversary_dual_restricted(
                nv,
                solver = solver,
                verbose = verbose,
                explicit_selected_full = selected,
                time_limit_sec = time_limit_sec,
            )
            row = (
                n = nv,
                value = result.value,
                status = string(result.status),
                orbits = length(result.selected_orbit_reps),
                inputs = result.localN,
                pair_orbits = result.n_pair_orbits,
                maxblock = maximum(b[2] for b in result.blocks),
                elapsed_sec = time() - t0,
            )
            append_csv_row(out_csv, row)
            push!(rows, row)
            write_simple_svg(out_svg, rows)
            println(
                "BALANCED_COMPLETE_BATCH n=$nv value=$(row.value) status=$(row.status) " *
                "orbits=$(row.orbits) inputs=$(row.inputs) pairs=$(row.pair_orbits) " *
                "maxblock=$(row.maxblock) elapsed_sec=$(round(row.elapsed_sec, digits=2))",
            )
        catch err
            println("FAILED n=$nv after $(round(time() - t0, digits=2)) seconds: $err")
            println("Stopping batch. Completed rows remain in $out_csv")
            break
        finally
            GC.gc()
        end
    end

    rows = read_completed_rows(out_csv)
    write_simple_svg(out_svg, rows)
    println("\nCompleted values")
    println("n value status orbits inputs pairs maxblock elapsed_sec")
    for row in sort(rows; by = r -> r.n)
        println("$(row.n) $(row.value) $(row.status) $(row.orbits) $(row.inputs) $(row.pair_orbits) $(row.maxblock) $(row.elapsed_sec)")
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    start_n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
    max_n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
    solver = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :mosek
    verbose = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : false
    time_limit_sec = length(ARGS) >= 5 ? parse_batch_time_limit(ARGS[5]) : 7200.0
    out_csv = length(ARGS) >= 6 ? ARGS[6] : joinpath(@__DIR__, "results", "balanced_complete_bipartite_values.csv")
    out_svg = length(ARGS) >= 7 ? ARGS[7] : joinpath(@__DIR__, "results", "balanced_complete_bipartite_values.svg")
    resume = length(ARGS) >= 8 ? parse(Bool, ARGS[8]) : true
    step = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 1
    run_balanced_complete_batch(
        start_n = start_n,
        max_n = max_n,
        step = step,
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
        out_csv = out_csv,
        out_svg = out_svg,
        resume = resume,
    )
end
