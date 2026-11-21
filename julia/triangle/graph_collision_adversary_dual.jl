# graph_collision_adversary_dual.jl
# Symmetry-reduced general adversary dual for graph collision.

using LinearAlgebra
using SparseArrays
using JuMP
using Clarabel
using PermutationGroups
using SymbolicWedderburn

const _MOSEKTOOLS_LOAD_ERROR_GRAPH_COLLISION = Ref{Any}(nothing)
try
    @eval import MosekTools
catch err
    _MOSEKTOOLS_LOAD_ERROR_GRAPH_COLLISION[] = err
end

include(joinpath(@__DIR__, "triangle_common.jl"))

function _make_graph_collision_optimizer(solver::Symbol)
    if solver == :clarabel
        return Clarabel.Optimizer
    elseif solver == :mosek
        if isdefined(Main, :MosekTools)
            return getproperty(Main.MosekTools, :Optimizer)
        end
        error("solver=:mosek requested, but MosekTools could not be loaded: " *
              "$(_MOSEKTOOLS_LOAD_ERROR_GRAPH_COLLISION[])")
    end
    error("Unknown solver $(solver). Use :clarabel or :mosek.")
end

function _gc_permutation(images::AbstractVector{<:Integer})
    degree = length(images)
    storage = degree <= typemax(UInt16) ? UInt16 :
              degree <= typemax(UInt32) ? UInt32 : UInt64
    return Perm{storage}(images)
end

function _normalize_host_edges(nv::Int, host_edges)
    edges = Tuple{Int, Int}[]
    for edge in host_edges
        u, v = edge
        1 <= u <= nv && 1 <= v <= nv || throw(ArgumentError("host edge $edge is out of range"))
        u != v || throw(ArgumentError("graph collision requires a loopless host graph"))
        u > v && ((u, v) = (v, u))
        push!(edges, (u, v))
    end
    sort!(unique!(edges))
    isempty(edges) && throw(ArgumentError("the host graph must contain at least one edge"))
    return edges
end

function _validate_automorphism_generators(nv::Int, host_edges, generators)
    edge_set = Set(host_edges)
    for sigma in generators
        sort(sigma) == collect(1:nv) || throw(ArgumentError("$sigma is not a permutation of 1:$nv"))
        mapped = Set{Tuple{Int, Int}}()
        for (u, v) in host_edges
            a, b = sigma[u], sigma[v]
            a > b && ((a, b) = (b, a))
            push!(mapped, (a, b))
        end
        mapped == edge_set || throw(ArgumentError("$sigma is not an automorphism of the host graph"))
    end
    return nothing
end

function graph_collision_function(nv::Int, host_edges)
    N = 1 << nv
    f = zeros(Int8, N)
    for bits in 0:(N - 1)
        for (u, v) in host_edges
            if ((bits >> (u - 1)) & 1) == 1 && ((bits >> (v - 1)) & 1) == 1
                f[bits + 1] = 1
                break
            end
        end
    end
    return f
end

function _vertex_perm_to_collision_input_perm(sigma::Vector{Int}, nv::Int)
    N = 1 << nv
    image = Vector{Int}(undef, N)
    for bits in 0:(N - 1)
        mapped = 0
        for vertex in 1:nv
            ((bits >> (vertex - 1)) & 1) == 1 || continue
            mapped |= 1 << (sigma[vertex] - 1)
        end
        image[bits + 1] = mapped + 1
    end
    return _gc_permutation(image)
end

function _gc_group_elements(nv::Int, vertex_generators)
    generators = [_gc_permutation(sigma) for sigma in vertex_generators]
    push!(generators, _gc_permutation(collect(1:nv)))
    return collect(PermGroup(generators))
end

function _query_orbits(nv::Int, vertex_group_elements)
    seen = falses(nv)
    orbits = Vector{Vector{Int}}()
    for vertex in 1:nv
        seen[vertex] && continue
        orbit = sort!(unique(Int(vertex^g) for g in vertex_group_elements))
        seen[orbit] .= true
        push!(orbits, orbit)
    end
    sort!(orbits; by = first)
    return orbits
end

function _inverse_images(permutation, degree::Int)
    inverse = Vector{Int}(undef, degree)
    for x in 1:degree
        inverse[Int(x^permutation)] = x
    end
    return inverse
end

function _gc_wedderburn_blocks(group, degree::Int; label::AbstractString, verbose::Bool)
    verbose && println("  Computing Wedderburn decomposition for $label...")
    start = time()
    sa_basis = SymbolicWedderburn.symmetry_adapted_basis(Float64, group; semisimple = false)
    verbose && println("  done in $(round(time() - start, digits=2))s")

    represented_degree = isempty(sa_basis) ? 0 : size(sa_basis[1].basis, 2)
    trailing_fixed = degree - represented_degree
    raw_blocks = NamedTuple[]
    for index in eachindex(sa_basis)
        block_size = size(sa_basis[index].basis, 1)
        block_size == 0 && continue
        rows, cols, values = findnz(sparse(sa_basis[index].basis))
        push!(raw_blocks, (
            index = index,
            U = sparse(rows, cols, values, block_size, degree),
            size = block_size,
            irrep_degree = SymbolicWedderburn.degree(sa_basis[index]),
            multiplicity = SymbolicWedderburn.multiplicity(sa_basis[index]),
        ))
    end

    if isempty(raw_blocks)
        push!(raw_blocks, (
            index = 1,
            U = sparse(1.0I, degree, degree),
            size = degree,
            irrep_degree = 1,
            multiplicity = degree,
        ))
        trailing_fixed = 0
    elseif trailing_fixed > 0
        trivial = findfirst(block -> block.irrep_degree == 1, raw_blocks)
        trivial === nothing && error("SymbolicWedderburn omitted fixed coordinates without a trivial block")
        old = raw_blocks[trivial]
        rows, cols, values = findnz(old.U)
        for k in 1:trailing_fixed
            push!(rows, old.size + k)
            push!(cols, represented_degree + k)
            push!(values, 1.0)
        end
        raw_blocks[trivial] = (
            index = old.index,
            U = sparse(rows, cols, values, old.size + trailing_fixed, degree),
            size = old.size + trailing_fixed,
            irrep_degree = old.irrep_degree,
            multiplicity = old.multiplicity + trailing_fixed,
        )
    end

    if verbose
        for block in raw_blocks
            println("    block $(block.index): PSD size=$(block.size), " *
                    "irrep_degree=$(block.irrep_degree), multiplicity=$(block.multiplicity)")
        end
    end
    return raw_blocks
end

function _gc_block_variables(model, raw_blocks, degree::Int, family::Int)
    blocks = NamedTuple[]
    for raw in raw_blocks
        B = @variable(
            model,
            [1:raw.size, 1:raw.size],
            PSD,
            base_name = "X_$(family)_$(raw.index)",
        )
        column_rows = [Int[] for _ in 1:degree]
        column_values = [Float64[] for _ in 1:degree]
        rows, cols, values = findnz(raw.U)
        for k in eachindex(values)
            push!(column_rows[cols[k]], rows[k])
            push!(column_values[cols[k]], values[k])
        end
        push!(blocks, (B = B, column_rows = column_rows, column_values = column_values))
    end
    return blocks
end

function _gc_invariant_entry(blocks, stabilizer_elements, degree::Int)
    images = [[Int(x^g) for x in 1:degree] for g in stabilizer_elements]
    inverse_order = 1.0 / length(images)
    cache = Dict{Tuple{Int, Int}, AffExpr}()

    function canonical_pair(x::Int, y::Int)
        x > y && ((x, y) = (y, x))
        best = (x, y)
        for image in images
            gx, gy = image[x], image[y]
            gx > gy && ((gx, gy) = (gy, gx))
            (gx, gy) < best && (best = (gx, gy))
        end
        return best
    end

    function entry(x::Int, y::Int)
        key = canonical_pair(x, y)
        haskey(cache, key) && return cache[key]
        terms = Dict{Tuple{Int, Int, Int}, Float64}()
        for image in images
            gx, gy = image[key[1]], image[key[2]]
            for (block_index, block) in enumerate(blocks)
                for ix in eachindex(block.column_rows[gx]), iy in eachindex(block.column_rows[gy])
                    coefficient = inverse_order *
                                  block.column_values[gx][ix] * block.column_values[gy][iy]
                    abs(coefficient) <= 1e-14 && continue
                    term = (
                        block_index,
                        block.column_rows[gx][ix],
                        block.column_rows[gy][iy],
                    )
                    terms[term] = get(terms, term, 0.0) + coefficient
                end
            end
        end
        expression = AffExpr(0.0)
        for ((block_index, row, column), coefficient) in terms
            abs(coefficient) > 1e-13 || continue
            add_to_expression!(expression, coefficient, blocks[block_index].B[row, column])
        end
        cache[key] = expression
        return expression
    end
    return entry, cache
end

"""
    graph_collision_adversary_dual(nv, host_edges, automorphism_generators; ...)

Solve the full general-adversary dual for graph collision on a fixed host
graph. An input marks a subset of host vertices, and the output is one iff a
host edge has both endpoints marked. `automorphism_generators` are vertex
permutations preserving `host_edges`.

The host automorphism group need not act transitively on queries. One canonical
PSD matrix is created for each vertex orbit and reduced by SymbolicWedderburn
under the stabilizer of that canonical vertex.
"""
function graph_collision_adversary_dual(
    nv::Int,
    host_edges,
    automorphism_generators;
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    nv >= 2 || throw(ArgumentError("graph collision requires at least two vertices"))
    nv < 8 * sizeof(Int) || throw(ArgumentError("too many vertices for bitstring indexing"))
    edges = _normalize_host_edges(nv, host_edges)
    vertex_generators = [collect(Int, sigma) for sigma in automorphism_generators]
    _validate_automorphism_generators(nv, edges, vertex_generators)

    N = 1 << nv
    f = graph_collision_function(nv, edges)
    vertex_elements = _gc_group_elements(nv, vertex_generators)
    input_generators = [
        _vertex_perm_to_collision_input_perm(sigma, nv) for sigma in vertex_generators
    ]
    push!(input_generators, _gc_permutation(collect(1:N)))
    input_group_elements = [
        _vertex_perm_to_collision_input_perm([Int(v^g) for v in 1:nv], nv) for g in vertex_elements
    ]
    query_orbits = _query_orbits(nv, vertex_elements)

    verbose && println("graph collision: nv=$nv, host_edges=$(length(edges)), inputs=$N")
    verbose && println("  automorphism group order: $(length(vertex_elements))")
    verbose && println("  query orbits: $query_orbits")

    # For every query, record its canonical family and the inverse input
    # transport taking it back to that family's representative.
    query_family = zeros(Int, nv)
    query_transport_inverse = Vector{Vector{Int}}(undef, nv)
    families = NamedTuple[]
    for (family, orbit) in enumerate(query_orbits)
        representative = first(orbit)
        stabilizer_indices = findall(g -> Int(representative^g) == representative, vertex_elements)
        stabilizer_inputs = input_group_elements[stabilizer_indices]
        H = PermGroup([stabilizer_inputs; _gc_permutation(collect(1:N))])
        H_elements = collect(H)
        raw_blocks = _gc_wedderburn_blocks(
            H,
            N;
            label = "query orbit $orbit (representative $representative, |H|=$(length(H_elements)))",
            verbose = verbose,
        )
        push!(families, (
            representative = representative,
            orbit = orbit,
            stabilizer_elements = H_elements,
            raw_blocks = raw_blocks,
        ))

        for query in orbit
            element_index = findfirst(g -> Int(representative^g) == query, vertex_elements)
            element_index === nothing && error("could not transport query $representative to $query")
            query_family[query] = family
            query_transport_inverse[query] = _inverse_images(input_group_elements[element_index], N)
        end
    end

    verbose && println("  Computing input and opposite-output pair orbits...")
    start = time()
    input_orbits, pair_orbits = compute_Sn_orbits(f, N, input_generators; store_pair_members = false)
    input_reps = sort!(collect(keys(input_orbits)))
    verbose && println("  done in $(round(time() - start, digits=2))s")
    verbose && println("    input orbits: $(length(input_reps))")
    verbose && println("    opposite-output pair orbits: $(length(pair_orbits))")

    optimizer = _make_graph_collision_optimizer(solver)
    model = solver == :mosek ? direct_model(optimizer()) : Model(optimizer)
    set_silent(model)
    solver == :mosek && set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
    time_limit_sec !== nothing && set_time_limit_sec(model, Float64(time_limit_sec))

    @variable(model, objective_bound)
    @objective(model, Min, 1.0 * objective_bound)

    family_models = NamedTuple[]
    block_details = Vector{Vector{NTuple{4, Int}}}()
    for (family, data) in enumerate(families)
        blocks = _gc_block_variables(model, data.raw_blocks, N, family)
        entry, cache = _gc_invariant_entry(blocks, data.stabilizer_elements, N)
        push!(family_models, (blocks = blocks, entry = entry, cache = cache))
        push!(block_details, [
            (block.index, block.size, block.irrep_degree, block.multiplicity)
            for block in data.raw_blocks
        ])
    end

    function query_matrix_entry(query::Int, x::Int, y::Int)
        family = query_family[query]
        inverse = query_transport_inverse[query]
        return family_models[family].entry(inverse[x], inverse[y])
    end

    verbose && println("  Adding $(length(pair_orbits)) crossing-pair constraints...")
    for (rep_zero, rep_one, _) in pair_orbits
        @assert f[rep_zero] != f[rep_one]
        zero_bits = rep_zero - 1
        one_bits = rep_one - 1
        expression = AffExpr(0.0)
        for query in 1:nv
            ((zero_bits >> (query - 1)) & 1) == ((one_bits >> (query - 1)) & 1) && continue
            expression += query_matrix_entry(query, rep_zero, rep_one)
        end
        @constraint(model, expression == 1.0)
    end

    verbose && println("  Adding $(length(input_reps)) diagonal-bound constraints...")
    for representative in input_reps
        expression = AffExpr(0.0)
        for query in 1:nv
            expression += query_matrix_entry(query, representative, representative)
        end
        @constraint(model, expression <= objective_bound)
    end

    n_input_orbits = length(input_reps)
    n_pair_orbits = length(pair_orbits)
    block_sizes = [[(index, size) for (index, size, _, _) in details] for details in block_details]
    cached_entry_orbits = [length(data.cache) for data in family_models]
    verbose && println("  reduced PSD sizes by query orbit: $block_sizes")
    verbose && println("  variables: $(num_variables(model)), constraints: " *
                       "$(num_constraints(model; count_variable_in_set_constraints = true))")

    # Release construction-only orbit tables before the solver starts.
    input_orbits = nothing
    pair_orbits = nothing
    input_reps = nothing
    families = nothing
    family_models = nothing
    input_group_elements = nothing
    vertex_elements = nothing
    GC.gc()
    try
        ccall(:malloc_trim, Cint, (Csize_t,), 0)
    catch
    end

    verbose && println("  Optimizing with $solver...")
    optimize!(model)
    status = termination_status(model)
    value = has_values(model) ? objective_value(model) : NaN
    verbose && println("  status: $status")
    verbose && println("  ADV±(graph collision) = $value")

    return (
        value = value,
        status = status,
        nv = nv,
        host_edges = edges,
        input_count = N,
        query_orbits = query_orbits,
        n_input_orbits = n_input_orbits,
        n_pair_orbits = n_pair_orbits,
        blocks = block_sizes,
        block_details = block_details,
        cached_entry_orbits = cached_entry_orbits,
        model = model,
    )
end

function _apply_vertex_perm_to_collision_bits(bits::Int, sigma::Vector{Int}, nv::Int)
    mapped = 0
    for vertex in 1:nv
        ((bits >> (vertex - 1)) & 1) == 1 || continue
        mapped |= 1 << (sigma[vertex] - 1)
    end
    return mapped
end

function _restricted_collision_input_perm(
    sigma::Vector{Int},
    selected_bits::Vector{Int},
    bit_to_local::Dict{Int, Int},
    nv::Int,
)
    images = Vector{Int}(undef, length(selected_bits))
    for (local_index, bits) in enumerate(selected_bits)
        mapped = _apply_vertex_perm_to_collision_bits(bits, sigma, nv)
        images[local_index] = get(bit_to_local, mapped, 0)
        images[local_index] != 0 || throw(ArgumentError(
            "selected graph-collision domain is not closed under automorphism $sigma",
        ))
    end
    return _gc_permutation(images)
end

"""
    graph_collision_adversary_dual_restricted(
        nv, host_edges, automorphism_generators, selected_bits; ...
    )

Solve the general-adversary dual on an automorphism-invariant promise of
marked subsets. `selected_bits` uses zero-based bit masks on the host vertices.
The returned value is a valid lower bound for graph collision on the full
domain.
"""
function graph_collision_adversary_dual_restricted(
    nv::Int,
    host_edges,
    automorphism_generators,
    selected_bits_input;
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    nv >= 2 || throw(ArgumentError("graph collision requires at least two vertices"))
    nv < 8 * sizeof(Int) || throw(ArgumentError("too many vertices for bitstring indexing"))
    edges = _normalize_host_edges(nv, host_edges)
    vertex_generators = [collect(Int, sigma) for sigma in automorphism_generators]
    _validate_automorphism_generators(nv, edges, vertex_generators)

    selected_bits = sort!(unique!(collect(Int, selected_bits_input)))
    isempty(selected_bits) && throw(ArgumentError("the selected promise is empty"))
    all(bits -> 0 <= bits < (1 << nv), selected_bits) ||
        throw(ArgumentError("a selected bit mask is outside 0:$(2^nv - 1)"))
    bit_to_local = Dict(bits => local_index for (local_index, bits) in enumerate(selected_bits))
    localN = length(selected_bits)
    f_full = graph_collision_function(nv, edges)
    f = Int8[f_full[bits + 1] for bits in selected_bits]
    any(==(Int8(0)), f) && any(==(Int8(1)), f) ||
        throw(ArgumentError("the selected promise must contain both outputs"))

    vertex_elements = _gc_group_elements(nv, vertex_generators)
    element_sigmas = [[Int(v^g) for v in 1:nv] for g in vertex_elements]
    input_group_elements = [
        _restricted_collision_input_perm(sigma, selected_bits, bit_to_local, nv)
        for sigma in element_sigmas
    ]
    input_generators = [
        _restricted_collision_input_perm(sigma, selected_bits, bit_to_local, nv)
        for sigma in vertex_generators
    ]
    push!(input_generators, _gc_permutation(collect(1:localN)))
    query_orbits = _query_orbits(nv, vertex_elements)

    verbose && println("restricted graph collision: nv=$nv inputs=$localN host_edges=$(length(edges))")
    verbose && println("  automorphism group order: $(length(vertex_elements))")
    verbose && println("  query orbits: $query_orbits")

    query_family = zeros(Int, nv)
    query_transport_inverse = Vector{Vector{Int}}(undef, nv)
    families = NamedTuple[]
    for (family, orbit) in enumerate(query_orbits)
        representative = first(orbit)
        stabilizer_indices = findall(g -> Int(representative^g) == representative, vertex_elements)
        stabilizer_inputs = input_group_elements[stabilizer_indices]
        H = PermGroup([stabilizer_inputs; _gc_permutation(collect(1:localN))])
        H_elements = collect(H)
        raw_blocks = _gc_wedderburn_blocks(
            H,
            localN;
            label = "restricted query orbit $orbit",
            verbose = verbose,
        )
        push!(families, (
            representative = representative,
            orbit = orbit,
            stabilizer_elements = H_elements,
            raw_blocks = raw_blocks,
        ))
        for query in orbit
            element_index = findfirst(g -> Int(representative^g) == query, vertex_elements)
            element_index === nothing && error("could not transport query $representative to $query")
            query_family[query] = family
            query_transport_inverse[query] =
                _inverse_images(input_group_elements[element_index], localN)
        end
    end

    input_orbits, pair_orbits =
        compute_Sn_orbits(f, localN, input_generators; store_pair_members = false)
    input_reps = sort!(collect(keys(input_orbits)))

    optimizer = _make_graph_collision_optimizer(solver)
    model = solver == :mosek ? direct_model(optimizer()) : Model(optimizer)
    set_silent(model)
    solver == :mosek && set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
    time_limit_sec !== nothing && set_time_limit_sec(model, Float64(time_limit_sec))

    @variable(model, objective_bound)
    @objective(model, Min, 1.0 * objective_bound)

    family_models = NamedTuple[]
    block_details = Vector{Vector{NTuple{4, Int}}}()
    for (family, data) in enumerate(families)
        blocks = _gc_block_variables(model, data.raw_blocks, localN, family)
        entry, cache = _gc_invariant_entry(blocks, data.stabilizer_elements, localN)
        push!(family_models, (blocks = blocks, entry = entry, cache = cache))
        push!(block_details, [
            (block.index, block.size, block.irrep_degree, block.multiplicity)
            for block in data.raw_blocks
        ])
    end

    function query_matrix_entry(query::Int, x::Int, y::Int)
        family = query_family[query]
        inverse = query_transport_inverse[query]
        return family_models[family].entry(inverse[x], inverse[y])
    end

    for (rep_zero, rep_one, _) in pair_orbits
        zero_bits = selected_bits[rep_zero]
        one_bits = selected_bits[rep_one]
        expression = AffExpr(0.0)
        for query in 1:nv
            ((zero_bits >> (query - 1)) & 1) == ((one_bits >> (query - 1)) & 1) && continue
            expression += query_matrix_entry(query, rep_zero, rep_one)
        end
        @constraint(model, expression == 1.0)
    end
    for representative in input_reps
        expression = AffExpr(0.0)
        for query in 1:nv
            expression += query_matrix_entry(query, representative, representative)
        end
        @constraint(model, expression <= objective_bound)
    end

    n_input_orbits = length(input_reps)
    n_pair_orbits = length(pair_orbits)
    block_sizes = [[(index, size) for (index, size, _, _) in details] for details in block_details]
    verbose && println("  input orbits=$n_input_orbits pair orbits=$n_pair_orbits blocks=$block_sizes")
    verbose && println("  Optimizing with $solver...")
    optimize!(model)
    status = termination_status(model)
    value = has_values(model) ? objective_value(model) : NaN
    verbose && println("  restricted ADV±(graph collision) = $value [$status]")

    return (
        value = value,
        status = status,
        nv = nv,
        host_edges = edges,
        selected_bits = selected_bits,
        input_count = localN,
        query_orbits = query_orbits,
        n_input_orbits = n_input_orbits,
        n_pair_orbits = n_pair_orbits,
        blocks = block_sizes,
        block_details = block_details,
        model = model,
    )
end

function star_graph_collision_adversary(
    nv::Int;
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
)
    nv >= 2 || throw(ArgumentError("a star requires at least two vertices"))
    host_edges = [(1, leaf) for leaf in 2:nv]
    automorphism_generators = Vector{Int}[]
    for leaf in 2:(nv - 1)
        sigma = collect(1:nv)
        sigma[leaf], sigma[leaf + 1] = sigma[leaf + 1], sigma[leaf]
        push!(automorphism_generators, sigma)
    end
    isempty(automorphism_generators) && push!(automorphism_generators, collect(1:nv))
    return graph_collision_adversary_dual(
        nv,
        host_edges,
        automorphism_generators;
        solver = solver,
        verbose = verbose,
        time_limit_sec = time_limit_sec,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    2 <= length(ARGS) <= 3 || error(
        "Usage: julia --project=. graph_collision_adversary_dual.jl star <nv> [solver=mosek]"
    )
    ARGS[1] == "star" || error("Only the built-in 'star' host is available from the CLI")
    nv = parse(Int, ARGS[2])
    solver = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :mosek
    result = star_graph_collision_adversary(nv; solver = solver, verbose = true)
    println(result)
end
