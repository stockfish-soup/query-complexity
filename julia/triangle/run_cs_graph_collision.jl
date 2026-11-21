include(joinpath(@__DIR__, "graph_collision_adversary_dual.jl"))

using Printf

"""
    cs_host(n)

Return the edge set and reflection generator of the graph CS(n). Vertices are
stored as `1:n`; this is the paper's labeling `0:n-1` shifted by one.
"""
function cs_host(n::Int)
    n >= 2 || throw(ArgumentError("CS(n) requires n >= 2"))
    square_offsets = [r^2 for r in 1:isqrt(n ÷ 2)]
    edges = [
        (u, u + offset)
        for offset in square_offsets
        for u in 1:(n - offset)
    ]
    reflection = [n + 1 - u for u in 1:n]
    return edges, [reflection], square_offsets
end

function _cs_max_block(result)
    return maximum(
        size
        for family in result.blocks
        for (_, size) in family
    )
end

function _cs_scalarized_psd(result)
    return sum(
        size * (size + 1) ÷ 2
        for family in result.blocks
        for (_, size) in family
    )
end

function _cs_existing_n(output_path::AbstractString)
    isfile(output_path) || return Set{Int}()
    solved = Set{Int}()
    for (line_number, line) in enumerate(eachline(output_path))
        line_number == 1 && continue
        fields = split(line, ',')
        isempty(fields) && continue
        value = tryparse(Int, fields[1])
        value === nothing || push!(solved, value)
    end
    return solved
end

function _cs_prepare_output(output_path::AbstractString; resume::Bool)
    if !resume || !isfile(output_path)
        open(output_path, "w") do io
            println(
                io,
                "n,offsets,host_edges,automorphisms,query_orbits,input_orbits," *
                "pair_orbits,max_block,scalarized_psd,value,sqrt_n,value_over_sqrt_n," *
                "status,elapsed_sec",
            )
        end
    end
end

function _cs_append_result(output_path::AbstractString, row)
    open(output_path, "a") do io
        println(
            io,
            join((
                row.n,
                join(row.offsets, ':'),
                row.host_edges,
                row.automorphisms,
                row.query_orbits,
                row.input_orbits,
                row.pair_orbits,
                row.max_block,
                row.scalarized_psd,
                @sprintf("%.12g", row.value),
                @sprintf("%.12g", row.sqrt_n),
                @sprintf("%.12g", row.value_over_sqrt_n),
                row.status,
                @sprintf("%.3f", row.elapsed_sec),
            ), ','),
        )
        flush(io)
    end
end

function run_cs_graph_collision_batch(
    first_n::Int,
    last_n::Int;
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
    output_path::AbstractString = joinpath(@__DIR__, "results", "cs_graph_collision_values.csv"),
    resume::Bool = true,
)
    2 <= first_n <= last_n || throw(ArgumentError("require 2 <= first_n <= last_n"))
    _cs_prepare_output(output_path; resume = resume)
    solved = resume ? _cs_existing_n(output_path) : Set{Int}()

    println("CS(n) graph-collision adversary batch")
    println("  n=$first_n:$last_n solver=$solver output=$output_path resume=$resume")
    for n in first_n:last_n
        if n in solved
            println("  n=$n already present; skipping")
            continue
        end

        edges, generators, offsets = cs_host(n)
        automorphisms = length(_gc_group_elements(n, generators))
        println("\nCS($n): offsets=$offsets edges=$(length(edges)) |Aut|=$automorphisms")
        started = time()
        result = graph_collision_adversary_dual(
            n,
            edges,
            generators;
            solver = solver,
            verbose = verbose,
            time_limit_sec = time_limit_sec,
        )
        elapsed = time() - started
        value = Float64(result.value)
        row = (
            n = n,
            offsets = offsets,
            host_edges = length(edges),
            automorphisms = automorphisms,
            query_orbits = length(result.query_orbits),
            input_orbits = result.n_input_orbits,
            pair_orbits = result.n_pair_orbits,
            max_block = _cs_max_block(result),
            scalarized_psd = _cs_scalarized_psd(result),
            value = value,
            sqrt_n = sqrt(n),
            value_over_sqrt_n = value / sqrt(n),
            status = string(result.status),
            elapsed_sec = elapsed,
        )
        _cs_append_result(output_path, row)
        println(
            "CS_RESULT n=$n value=$(row.value) value/sqrt(n)=$(row.value_over_sqrt_n) " *
            "status=$(row.status) input_orbits=$(row.input_orbits) " *
            "pair_orbits=$(row.pair_orbits) maxblock=$(row.max_block) elapsed=$(round(elapsed, digits=2))s",
        )
        result = nothing
        GC.gc()
        try
            ccall(:malloc_trim, Cint, (Csize_t,), 0)
        catch
        end
    end
    println("\nCheckpointed results to $output_path")
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    0 <= length(ARGS) <= 7 || error(
        "Usage: julia --project=. run_cs_graph_collision.jl " *
        "[first_n=2] [last_n=9] [solver=mosek] [verbose=true] " *
        "[time_limit_sec=none] [output=cs_graph_collision_values.csv] [resume=true]",
    )
    first_n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
    last_n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 9
    solver = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :mosek
    verbose = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : true
    time_limit = length(ARGS) >= 5 && ARGS[5] != "none" ? parse(Float64, ARGS[5]) : nothing
    output = length(ARGS) >= 6 ? ARGS[6] : joinpath(@__DIR__, "results", "cs_graph_collision_values.csv")
    resume = length(ARGS) >= 7 ? parse(Bool, ARGS[7]) : true
    run_cs_graph_collision_batch(
        first_n,
        last_n;
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit,
        output_path = output,
        resume = resume,
    )
end
