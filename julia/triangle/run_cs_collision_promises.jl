include(joinpath(@__DIR__, "run_cs_graph_collision.jl"))

using Printf

function _cs_reflect_bits(bits::Int, n::Int)
    return _apply_vertex_perm_to_collision_bits(bits, [n + 1 - v for v in 1:n], n)
end

_cs_canonical_bits(bits::Int, n::Int) = min(bits, _cs_reflect_bits(bits, n))

function _cs_marked_vertices(bits::Int, n::Int)
    return [v - 1 for v in 1:n if ((bits >> (v - 1)) & 1) == 1]
end

function _cs_positive_completions(bits::Int, n::Int, f::Vector{Int8})
    positives = Int[]
    for vertex in 1:n
        ((bits >> (vertex - 1)) & 1) == 0 || continue
        completed = bits | (1 << (vertex - 1))
        f[completed + 1] == 1 && push!(positives, _cs_canonical_bits(completed, n))
    end
    return sort!(unique!(positives))
end

function _cs_expand_reps(reps, n::Int)
    selected = Set{Int}()
    for bits in reps
        push!(selected, bits)
        push!(selected, _cs_reflect_bits(bits, n))
    end
    return sort!(collect(selected))
end

function cs_negative_orbit_metadata(n::Int)
    edges, _, offsets = cs_host(n)
    f = graph_collision_function(n, edges)
    seen = Set{Int}()
    rows = NamedTuple[]
    for bits in 0:((1 << n) - 1)
        f[bits + 1] == 0 || continue
        representative = _cs_canonical_bits(bits, n)
        representative in seen && continue
        push!(seen, representative)
        completions = _cs_positive_completions(representative, n, f)
        isempty(completions) && continue
        reflected = _cs_reflect_bits(representative, n)
        marked = _cs_marked_vertices(representative, n)
        push!(rows, (
            rep = representative,
            orbit_size = reflected == representative ? 1 : 2,
            stabilizer = reflected == representative ? 2 : 1,
            marked_count = length(marked),
            marked = marked,
            completing = length([
                vertex for vertex in 1:n
                if ((representative >> (vertex - 1)) & 1) == 0 &&
                   f[(representative | (1 << (vertex - 1))) + 1] == 1
            ]),
            positive_reps = completions,
            offsets = offsets,
        ))
    end
    sort!(rows; by = row -> (-row.completing, -row.stabilizer, row.marked_count, row.rep))
    return rows
end

function _cs_promise_reps(negative_reps, metadata_by_rep)
    reps = Set{Int}()
    for negative in negative_reps
        push!(reps, negative)
        union!(reps, metadata_by_rep[negative].positive_reps)
    end
    return sort!(collect(reps))
end

function _cs_solve_promise(n, edges, generators, negative_reps, metadata_by_rep; solver, verbose)
    reps = _cs_promise_reps(negative_reps, metadata_by_rep)
    selected = _cs_expand_reps(reps, n)
    result = graph_collision_adversary_dual_restricted(
        n,
        edges,
        generators,
        selected;
        solver = solver,
        verbose = verbose,
    )
    maxblock = maximum(size for family in result.blocks for (_, size) in family)
    return result, reps, maxblock
end

function _cs_write_csv(path, rows)
    open(path, "w") do io
        println(io, "round,rep,marked,marked_count,c,orbit_size,positive_reps,value,fraction_full,inputs,input_orbits,pair_orbits,maxblock")
        for row in rows
            println(io, join((
                row.round,
                row.rep,
                join(row.marked, ':'),
                row.marked_count,
                row.completing,
                row.orbit_size,
                join(row.positive_reps, ':'),
                @sprintf("%.12g", row.value),
                @sprintf("%.12g", row.fraction_full),
                row.inputs,
                row.input_orbits,
                row.pair_orbits,
                row.maxblock,
            ), ','))
        end
    end
end

function run_cs_collision_promise_search(
    n::Int;
    rounds::Int = 0,
    solver::Symbol = :mosek,
    verbose::Bool = false,
    full_value::Union{Nothing, Float64} = nothing,
    output_path::AbstractString = joinpath(@__DIR__, "results", "cs_collision_promise_n$(n).csv"),
)
    edges, generators, _ = cs_host(n)
    metadata = cs_negative_orbit_metadata(n)
    metadata_by_rep = Dict(row.rep => row for row in metadata)
    rounds <= 0 && (rounds = length(metadata))

    if full_value === nothing
        full = graph_collision_adversary_dual(n, edges, generators; solver = solver, verbose = verbose)
        full_value = Float64(full.value)
        full = nothing
        GC.gc()
    end
    println("CS($n) full adversary value: $full_value")
    println("negative independent-set orbits with c(S)>0: $(length(metadata))")

    # First compare every negative orbit in isolation. This provides a cheap,
    # interpretable importance score before cumulative selection.
    standalone = NamedTuple[]
    println("\nStandalone negative-orbit promises")
    println("rank rep marked c orbit positives value fraction inputs")
    for row in metadata
        result, _, maxblock = _cs_solve_promise(
            n, edges, generators, [row.rep], metadata_by_rep;
            solver = solver, verbose = false,
        )
        entry = merge(row, (
            value = Float64(result.value),
            fraction_full = Float64(result.value) / full_value,
            inputs = result.input_count,
            input_orbits = result.n_input_orbits,
            pair_orbits = result.n_pair_orbits,
            maxblock = maxblock,
        ))
        push!(standalone, entry)
        result = nothing
        GC.gc()
    end
    sort!(standalone; by = row -> (-row.value, row.inputs, -row.completing, row.rep))
    for (rank, row) in enumerate(standalone)
        println("$rank $(row.rep) $(row.marked) $(row.completing) $(row.orbit_size) " *
                "$(row.positive_reps) $(row.value) $(row.fraction_full) $(row.inputs)")
    end

    # Add orbits by standalone adversary value. Ties favor fewer explicit
    # inputs, then larger one-query boundary c(S).
    selected_negatives = Int[]
    cumulative = NamedTuple[]
    println("\nCumulative promise")
    for (round, row) in enumerate(Iterators.take(standalone, rounds))
        push!(selected_negatives, row.rep)
        result, reps, maxblock = _cs_solve_promise(
            n, edges, generators, selected_negatives, metadata_by_rep;
            solver = solver, verbose = verbose,
        )
        entry = merge(row, (
            round = round,
            value = Float64(result.value),
            fraction_full = Float64(result.value) / full_value,
            inputs = result.input_count,
            input_orbits = result.n_input_orbits,
            pair_orbits = result.n_pair_orbits,
            maxblock = maxblock,
            promise_reps = reps,
        ))
        push!(cumulative, entry)
        println("PROMISE round=$round rep=$(row.rep) marked=$(row.marked) c=$(row.completing) " *
                "value=$(entry.value) fraction=$(entry.fraction_full) inputs=$(entry.inputs) " *
                "orbits=$(entry.input_orbits) pairs=$(entry.pair_orbits) maxblock=$maxblock")
        result = nothing
        GC.gc()
    end
    _cs_write_csv(output_path, cumulative)
    println("Wrote $output_path")
    return (full_value = full_value, standalone = standalone, cumulative = cumulative)
end

function run_cs_collision_marginal_search(
    n::Int,
    rounds::Int;
    solver::Symbol = :mosek,
    verbose::Bool = false,
    full_value::Float64,
    output_path::AbstractString = joinpath(@__DIR__, "results", "cs_collision_marginal_n$(n).csv"),
)
    rounds >= 1 || throw(ArgumentError("rounds must be positive"))
    edges, generators, _ = cs_host(n)
    metadata = cs_negative_orbit_metadata(n)
    metadata_by_rep = Dict(row.rep => row for row in metadata)
    remaining = Set(row.rep for row in metadata)
    selected = Int[]
    results = NamedTuple[]

    println("CS($n) marginal promise search: candidates=$(length(remaining)) rounds=$rounds")
    for round in 1:min(rounds, length(remaining))
        trials = NamedTuple[]
        for candidate in sort!(collect(remaining))
            negative_reps = [selected; candidate]
            result, _, maxblock = _cs_solve_promise(
                n, edges, generators, negative_reps, metadata_by_rep;
                solver = solver, verbose = false,
            )
            row = metadata_by_rep[candidate]
            push!(trials, merge(row, (
                value = Float64(result.value),
                fraction_full = Float64(result.value) / full_value,
                inputs = result.input_count,
                input_orbits = result.n_input_orbits,
                pair_orbits = result.n_pair_orbits,
                maxblock = maxblock,
            )))
            result = nothing
            GC.gc()
        end
        sort!(trials; by = row -> (-row.value, row.inputs, -row.completing, row.rep))
        best = first(trials)
        push!(selected, best.rep)
        delete!(remaining, best.rep)
        entry = merge(best, (round = round,))
        push!(results, entry)
        println("MARGINAL round=$round rep=$(best.rep) marked=$(best.marked) c=$(best.completing) " *
                "value=$(best.value) fraction=$(best.fraction_full) inputs=$(best.inputs) " *
                "pairs=$(best.pair_orbits) maxblock=$(best.maxblock)")
    end
    _cs_write_csv(output_path, results)
    println("Wrote $output_path")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS) && ARGS[1] == "marginal"
        3 <= length(ARGS) <= 7 || error(
            "Usage: julia --project=. run_cs_collision_promises.jl marginal <n> <rounds> " *
            "[solver=mosek] [verbose=false] <full_value> [output]",
        )
        n = parse(Int, ARGS[2])
        rounds = parse(Int, ARGS[3])
        solver = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :mosek
        verbose = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : false
        length(ARGS) >= 6 || error("marginal mode requires the known full adversary value")
        full_value = parse(Float64, ARGS[6])
        output = length(ARGS) >= 7 ? ARGS[7] : joinpath(@__DIR__, "results", "cs_collision_marginal_n$(n).csv")
        run_cs_collision_marginal_search(
            n,
            rounds;
            solver = solver,
            verbose = verbose,
            full_value = full_value,
            output_path = output,
        )
    else
        1 <= length(ARGS) <= 6 || error(
            "Usage: julia --project=. run_cs_collision_promises.jl <n> [rounds=0/all] " *
            "[solver=mosek] [verbose=false] [full_value=compute] [output]",
        )
        n = parse(Int, ARGS[1])
        rounds = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
        solver = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :mosek
        verbose = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : false
        full_value = length(ARGS) >= 5 && ARGS[5] != "compute" ? parse(Float64, ARGS[5]) : nothing
        output = length(ARGS) >= 6 ? ARGS[6] : joinpath(@__DIR__, "results", "cs_collision_promise_n$(n).csv")
        run_cs_collision_promise_search(
            n;
            rounds = rounds,
            solver = solver,
            verbose = verbose,
            full_value = full_value,
            output_path = output,
        )
    end
end
