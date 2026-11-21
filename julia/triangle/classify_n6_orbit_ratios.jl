include(joinpath(@__DIR__, "run_triangle_symmetric.jl"))

function n6_candidate_orbits(; min_completing::Int = 1)
    nv = 6
    meta, E, edge_index, _, _ = orbit_metadata_by_rep(nv)
    candidates = [
        d for d in values(meta)
        if d.output == 0 && d.completing_edges >= min_completing
    ]
    sort!(candidates; by = d -> (-d.completing_edges, d.edges, d.orbit_size, d.rep))
    return candidates, E, edge_index
end

function single_negative_promise_reps(nv::Int, neg_rep::Integer, E, edge_index)
    fullN = big(1) << binomial(nv, 2)
    reps = BigInt[big(1), fullN, BigInt(neg_rep)]
    for edge_idx in completing_edge_indices(nv, neg_rep, E, edge_index)
        push!(reps, BigInt(canonical_graph_rep(nv, add_edge_rep(neg_rep, edge_idx))))
    end
    return sort!(unique!(reps))
end

function solve_single_orbit_ratio(
    d,
    E,
    edge_index;
    solver::Symbol = :mosek,
    verbose::Bool = false,
    max_pair_members::Int = 2_000_000,
)
    nv = 6
    reps = single_negative_promise_reps(nv, d.rep, E, edge_index)
    general = solve_triangle_with_orbits(
        nv,
        reps;
        solver = solver,
        verbose = verbose,
        time_limit_sec = nothing,
    )
    positive = triangle_adversary_primal_restricted(
        nv,
        Int.(reps);
        solver = solver,
        verbose = verbose,
        max_pair_members = max_pair_members,
        positive_gamma = true,
    )
    return (
        rep = d.rep,
        edges = d.edges,
        completing = d.completing_edges,
        orbit_size = d.orbit_size,
        automorphisms = d.automorphisms,
        degrees = join(d.degrees, ":"),
        positives = join(setdiff(reps, BigInt[big(1), big(1) << binomial(nv, 2), BigInt(d.rep)]), ":"),
        general = general.value,
        positive = positive.value,
        ratio = general.value / positive.value,
        status_general = string(general.status),
        status_positive = string(positive.status),
        orbits = length(reps),
        inputs = general.localN,
        pair_orbits = general.n_pair_orbits,
        pair_members = positive.pair_members,
        maxblock = maximum(b[2] for b in general.blocks),
    )
end

function write_ratio_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "rank,rep,edges,c,orbit,aut,degrees,positives,general,positive,ratio,status_general,status_positive,orbits,inputs,pair_orbits,pair_members,maxblock")
        sorted = sort(rows; by = r -> r.ratio, rev = true)
        for (rank, r) in enumerate(sorted)
            println(io, join((
                rank, r.rep, r.edges, r.completing, r.orbit_size, r.automorphisms,
                r.degrees, r.positives, r.general, r.positive, r.ratio,
                r.status_general, r.status_positive, r.orbits, r.inputs,
                r.pair_orbits, r.pair_members, r.maxblock,
            ), ","))
        end
    end
end

function plot_ratio_svg(path::AbstractString, rows)
    sorted = sort(rows; by = r -> r.ratio, rev = true)
    isempty(sorted) && return nothing
    width = 1000
    height = 520
    left = 70
    right = 20
    top = 34
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_ratio = maximum(r.ratio for r in sorted) * 1.05
    min_ratio = 1.0
    xcoord(i) = left + (i - 1) / max(1, length(sorted) - 1) * plot_w
    ycoord(v) = top + (1 - (v - min_ratio) / (max_ratio - min_ratio)) * plot_h
    points = join(["$(round(xcoord(i), digits=2)),$(round(ycoord(r.ratio), digits=2))" for (i, r) in enumerate(sorted)], " ")

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$(width/2)" y="22" text-anchor="middle" font-family="sans-serif" font-size="16">n=6 single-orbit promises ranked by Adv± / Adv+</text>""")
        println(io, """<line x1="$left" y1="$(height-bottom)" x2="$(width-right)" y2="$(height-bottom)" stroke="#111827"/>""")
        println(io, """<line x1="$left" y1="$top" x2="$left" y2="$(height-bottom)" stroke="#111827"/>""")
        println(io, """<polyline points="$points" fill="none" stroke="#2563eb" stroke-width="2"/>""")
        for (i, r) in enumerate(sorted)
            x = xcoord(i)
            y = ycoord(r.ratio)
            println(io, """<circle cx="$x" cy="$y" r="4" fill="#2563eb"/>""")
            if i <= 12
                println(io, """<text x="$x" y="$(y-9)" text-anchor="middle" font-family="sans-serif" font-size="9">$(r.rep)</text>""")
            end
        end
        println(io, """<text x="$(width/2)" y="$(height-18)" text-anchor="middle" font-family="sans-serif" font-size="13">orbit rank by ratio</text>""")
        println(io, """<text x="18" y="$(height/2)" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 18 $(height/2))">Adv± / Adv+</text>""")
        println(io, "</svg>")
    end
end

function classify_n6_orbit_ratios(;
    limit::Int = 0,
    start_index::Int = 1,
    skip_reps::Set{Int} = Set{Int}(),
    solver::Symbol = :mosek,
    verbose::Bool = false,
    max_pair_members::Int = 2_000_000,
    out_csv::AbstractString = joinpath(@__DIR__, "results", "n6_single_orbit_ratios.csv"),
    out_svg::AbstractString = joinpath(@__DIR__, "results", "n6_single_orbit_ratios.svg"),
)
    candidates, E, edge_index = n6_candidate_orbits()
    limit > 0 && (candidates = candidates[1:min(limit, length(candidates))])
    start_index > 1 && (candidates = candidates[start_index:end])
    !isempty(skip_reps) && filter!(d -> !(Int(d.rep) in skip_reps), candidates)
    println("n=6 candidate negative orbits with c(G)>=1: $(length(candidates))")
    println("order rep edges c orbit aut degrees")
    for (i, d) in enumerate(candidates)
        println("$i $(d.rep) $(d.edges) $(d.completing_edges) $(d.orbit_size) $(d.automorphisms) $(join(d.degrees, ':'))")
    end

    rows = NamedTuple[]
    for (i, d) in enumerate(candidates)
        println("\n[$i/$(length(candidates))] rep=$(d.rep), edges=$(d.edges), c=$(d.completing_edges), degrees=$(join(d.degrees, ':'))")
        try
            row = solve_single_orbit_ratio(
                d,
                E,
                edge_index;
                solver = solver,
                verbose = verbose,
                max_pair_members = max_pair_members,
            )
            push!(rows, row)
            write_ratio_csv(out_csv, rows)
            plot_ratio_svg(out_svg, rows)
            println(
                "ORBIT_RATIO rep=$(row.rep) c=$(row.completing) edges=$(row.edges) " *
                "general=$(row.general) positive=$(row.positive) ratio=$(row.ratio) " *
                "inputs=$(row.inputs) pair_members=$(row.pair_members)",
            )
        catch err
            println("  failed: $err")
        end
        GC.gc()
    end

    sorted = sort(rows; by = r -> r.ratio, rev = true)
    println("\nTop ratios")
    println("rank rep edges c orbit aut degrees general positive ratio positives")
    for (rank, r) in enumerate(Iterators.take(sorted, 20))
        println("$(rank) $(r.rep) $(r.edges) $(r.completing) $(r.orbit_size) $(r.automorphisms) $(r.degrees) $(r.general) $(r.positive) $(r.ratio) $(r.positives)")
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    limit = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
    solver = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :mosek
    verbose = length(ARGS) >= 3 ? parse(Bool, ARGS[3]) : false
    max_pair_members = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 2_000_000
    start_index = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
    skip_reps = if length(ARGS) >= 6 && ARGS[6] != "none"
        Set(parse.(Int, split(ARGS[6], ",")))
    else
        Set{Int}()
    end
    out_csv = length(ARGS) >= 7 ? ARGS[7] : joinpath(@__DIR__, "results", "n6_single_orbit_ratios.csv")
    out_svg = length(ARGS) >= 8 ? ARGS[8] : joinpath(@__DIR__, "results", "n6_single_orbit_ratios.svg")
    classify_n6_orbit_ratios(
        limit = limit,
        start_index = start_index,
        skip_reps = skip_reps,
        solver = solver,
        verbose = verbose,
        max_pair_members = max_pair_members,
        out_csv = out_csv,
        out_svg = out_svg,
    )
end
