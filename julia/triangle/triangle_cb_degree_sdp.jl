# triangle_cb_degree_sdp.jl
# Symmetry-reduced implementation of the completely bounded degree SDP (18)
# from Gribling--Laurent, arXiv:1901.04921v1, specialized to triangle finding.

using LinearAlgebra
using SparseArrays
using JuMP
using Clarabel
using PermutationGroups
using SymbolicWedderburn

const _MOSEKTOOLS_LOAD_ERROR_CB = Ref{Any}(nothing)
try
    @eval import MosekTools
catch err
    _MOSEKTOOLS_LOAD_ERROR_CB[] = err
end

include(joinpath(@__DIR__, "triangle_common.jl"))

function _make_cb_optimizer(solver::Symbol)
    if solver == :clarabel
        return Clarabel.Optimizer
    elseif solver == :mosek
        if isdefined(Main, :MosekTools)
            return getproperty(Main.MosekTools, :Optimizer)
        end
        error("solver=:mosek requested, but MosekTools could not be loaded: $(_MOSEKTOOLS_LOAD_ERROR_CB[])")
    end
    error("Unknown solver $(solver). Use :clarabel or :mosek.")
end

function _checked_power(base::Int, exponent::Int, description::AbstractString)
    exponent >= 0 || throw(ArgumentError("$description exponent must be nonnegative"))
    value = big(base)^exponent
    value <= typemax(Int) || error("$description is too large to index with Int: $base^$exponent")
    return Int(value)
end

"""Construct a permutation using an image type wide enough for its degree."""
function _permutation(images::AbstractVector{<:Integer})
    degree = length(images)
    storage = if degree <= typemax(UInt16)
        UInt16
    elseif degree <= typemax(UInt32)
        UInt32
    else
        UInt64
    end
    return Perm{storage}(images)
end

function _vertex_label_generators(nv::Int, E, edge_index)
    m = length(E)
    q = m + 1
    generators = Vector{Int}[]
    for vertex in 1:(nv - 1)
        sigma = collect(1:nv)
        sigma[vertex], sigma[vertex + 1] = sigma[vertex + 1], sigma[vertex]
        label_perm = collect(1:q)
        for (label, (u, v)) in enumerate(E)
            a, b = sigma[u], sigma[v]
            a > b && ((a, b) = (b, a))
            label_perm[label] = edge_index[(a, b)]
        end
        # The last coordinate is the homogenizing coordinate in z=(x,1).
        label_perm[q] = q
        push!(generators, label_perm)
    end
    return generators
end

"""Permutation of q-ary k-tuples, encoded little-endian as integers 0:q^k-1."""
function _tuple_permutation(label_perm::Vector{Int}, k::Int)
    q = length(label_perm)
    tuple_count = _checked_power(q, k, "tuple space")
    image = Vector{Int}(undef, tuple_count)
    for code in 0:(tuple_count - 1)
        rest = code
        mapped = 0
        place = 1
        for _ in 1:k
            digit = (rest % q) + 1
            rest ÷= q
            mapped += (label_perm[digit] - 1) * place
            place *= q
        end
        image[code + 1] = mapped + 1
    end
    return image
end

function _reverse_tuple_code(code::Int, q::Int, k::Int)
    reversed = 0
    for _ in 1:k
        reversed = q * reversed + (code % q)
        code ÷= q
    end
    return reversed
end

function _tuple_actions(
    label_generators,
    q::Int,
    s::Int,
    r::Int;
    reversal_symmetry::Bool = false,
)
    left_count = _checked_power(q, s, "left tuple space")
    right_count = _checked_power(q, r, "right tuple space")
    degree = left_count + right_count
    generators = Perm[]
    for label_perm in label_generators
        left_image = _tuple_permutation(label_perm, s)
        right_image = _tuple_permutation(label_perm, r)
        image = Vector{Int}(undef, degree)
        image[1:left_count] .= left_image
        image[(left_count + 1):degree] .= left_count .+ right_image
        push!(generators, _permutation(image))
    end
    if reversal_symmetry
        s == r || throw(ArgumentError("matrix reversal symmetry requires s == t-s"))
        image = Vector{Int}(undef, degree)
        for code in 0:(left_count - 1)
            reversed = _reverse_tuple_code(code, q, s)
            image[code + 1] = left_count + reversed + 1
            image[left_count + code + 1] = reversed + 1
        end
        push!(generators, _permutation(image))
    end
    isempty(generators) && push!(generators, _permutation(collect(1:degree)))
    return generators, left_count, right_count
end

function _full_tuple_actions(
    label_generators,
    q::Int,
    t::Int;
    reversal_symmetry::Bool = false,
)
    tuple_count = _checked_power(q, t, "tensor")
    generators = Perm[]
    for label_perm in label_generators
        push!(generators, _permutation(_tuple_permutation(label_perm, t)))
    end
    if reversal_symmetry
        image = [_reverse_tuple_code(code, q, t) + 1 for code in 0:(tuple_count - 1)]
        push!(generators, _permutation(image))
    end
    isempty(generators) && push!(generators, _permutation(collect(1:tuple_count)))
    return generators, tuple_count
end

mutable struct _UnionFind
    parent::Vector{Int}
    rank::Vector{UInt8}
end

_UnionFind(n::Int) = _UnionFind(collect(1:n), zeros(UInt8, n))

function _find!(uf::_UnionFind, x::Int)
    while uf.parent[x] != x
        uf.parent[x] = uf.parent[uf.parent[x]]
        x = uf.parent[x]
    end
    return x
end

function _union!(uf::_UnionFind, x::Int, y::Int)
    rx = _find!(uf, x)
    ry = _find!(uf, y)
    rx == ry && return rx
    if uf.rank[rx] < uf.rank[ry]
        rx, ry = ry, rx
    end
    uf.parent[ry] = rx
    uf.rank[rx] == uf.rank[ry] && (uf.rank[rx] += 1)
    return rx
end

function _orbits_from_permutations(degree::Int, generators)
    uf = _UnionFind(degree)
    for g in generators, x in 1:degree
        _union!(uf, x, Int(x^g))
    end
    members_by_root = Dict{Int, Vector{Int}}()
    for x in 1:degree
        push!(get!(members_by_root, _find!(uf, x), Int[]), x)
    end
    members = sort!(collect(values(members_by_root)); by = first)
    orbit_of = zeros(Int, degree)
    for (orbit, xs) in enumerate(members), x in xs
        orbit_of[x] = orbit
    end
    return orbit_of, members
end

function _graph_orbit_representatives(nv::Int, f, E, edge_index)
    m = length(E)
    graph_count = length(f)
    generators = full_Sn_generators(nv, E, edge_index, m, graph_count)
    _, orbits = _orbits_from_permutations(graph_count, generators)
    return first.(orbits)
end

"""
Build the connected components of matrix entries identified by (7) and (8).

Each row of A_s is an equality between two off-diagonal Gram entries.  If those
entries are regarded as vertices of a graph, range(A_s^*) is the incidence
space of that graph: its elements sum to zero on every connected component.
This lets the SDP eliminate y from (18) exactly.
"""
function _as_entry_components(q::Int, s::Int, r::Int, left_count::Int, right_count::Int)
    pairs = Tuple{Int, Int}[]
    for block in ((0, left_count), (left_count, right_count))
        offset, count = block
        for x in 1:count, y in (x + 1):count
            push!(pairs, (offset + x, offset + y))
        end
    end
    pair_index = Dict(pair => i for (i, pair) in enumerate(pairs))
    uf = _UnionFind(length(pairs))

    # Equation (8): u_{j i}, with i a suffix of length ell.
    for ell in 1:(s - 1)
        prefix_count = _checked_power(q, s - ell, "equation (8) prefix")
        suffix_count = _checked_power(q, ell, "equation (8) suffix")
        for j in 0:(prefix_count - 1), k in (j + 1):(prefix_count - 1)
            base = (j + 1, k + 1)
            base_index = pair_index[base]
            for suffix in 1:(suffix_count - 1)
                a = j + prefix_count * suffix + 1
                b = k + prefix_count * suffix + 1
                _union!(uf, base_index, pair_index[(a, b)])
            end
        end
    end

    # Equation (7): v_{i j}, with i a prefix of length ell.
    for ell in 1:(r - 1)
        prefix_count = _checked_power(q, ell, "equation (7) prefix")
        suffix_count = _checked_power(q, r - ell, "equation (7) suffix")
        for j in 0:(suffix_count - 1), k in (j + 1):(suffix_count - 1)
            base_a = left_count + prefix_count * j + 1
            base_b = left_count + prefix_count * k + 1
            base_index = pair_index[(base_a, base_b)]
            for prefix in 1:(prefix_count - 1)
                a = left_count + prefix + prefix_count * j + 1
                b = left_count + prefix + prefix_count * k + 1
                _union!(uf, base_index, pair_index[(a, b)])
            end
        end
    end

    entries_by_root = Dict{Int, Vector{Tuple{Int, Int}}}()
    for (index, pair) in enumerate(pairs)
        push!(get!(entries_by_root, _find!(uf, index), Tuple{Int, Int}[]), pair)
    end
    components = collect(values(entries_by_root))
    sort!(components; by = component -> first(component))
    component_of_pair = Dict{Tuple{Int, Int}, Int}()
    for (component, entries) in enumerate(components), pair in entries
        component_of_pair[pair] = component
    end
    return components, component_of_pair
end

function _component_orbit_representatives(components, component_of_pair, generators)
    uf = _UnionFind(length(components))
    for (component, entries) in enumerate(components)
        x, y = first(entries)
        for g in generators
            gx, gy = Int(x^g), Int(y^g)
            gx > gy && ((gx, gy) = (gy, gx))
            _union!(uf, component, component_of_pair[(gx, gy)])
        end
    end
    representative = Dict{Int, Int}()
    for component in eachindex(components)
        root = _find!(uf, component)
        representative[root] = min(get(representative, root, component), component)
    end
    return sort!(collect(values(representative)))
end

function _wedderburn_blocks(group, degree::Int; verbose::Bool)
    verbose && println("  Computing Wedderburn decomposition...")
    start = time()
    sa_basis = SymbolicWedderburn.symmetry_adapted_basis(Float64, group; semisimple = false)
    verbose && println("  done in $(round(time() - start, digits=2))s")

    represented_degree = isempty(sa_basis) ? 0 : size(sa_basis[1].basis, 2)
    trailing_fixed = degree - represented_degree
    raw_blocks = NamedTuple[]
    for index in eachindex(sa_basis)
        size(sa_basis[index].basis, 1) == 0 && continue
        rows, cols, vals = findnz(sparse(sa_basis[index].basis))
        block_size = size(sa_basis[index].basis, 1)
        push!(raw_blocks, (
            index = index,
            U = sparse(rows, cols, vals, block_size, degree),
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
        trivial === nothing && error("SymbolicWedderburn omitted fixed coordinates but returned no trivial block")
        old = raw_blocks[trivial]
        rows, cols, vals = findnz(old.U)
        for k in 1:trailing_fixed
            push!(rows, old.size + k)
            push!(cols, represented_degree + k)
            push!(vals, 1.0)
        end
        raw_blocks[trivial] = (
            index = old.index,
            U = sparse(rows, cols, vals, old.size + trailing_fixed, degree),
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
        represented = sum(block.irrep_degree * block.multiplicity for block in raw_blocks)
        println("    represented dimensions: $represented (should be $degree)")
    end
    return raw_blocks
end

function _block_variables(model, raw_blocks, degree::Int)
    blocks = NamedTuple[]
    for raw in raw_blocks
        B = @variable(model, [1:raw.size, 1:raw.size], PSD, base_name = "W_$(raw.index)")
        column_rows = [Int[] for _ in 1:degree]
        column_values = [Float64[] for _ in 1:degree]
        rows, cols, vals = findnz(raw.U)
        for k in eachindex(vals)
            push!(column_rows[cols[k]], rows[k])
            push!(column_values[cols[k]], vals[k])
        end
        push!(blocks, (B = B, column_rows = column_rows, column_values = column_values))
    end
    return blocks
end

function _invariant_psd_entry_function(blocks, group_elements, degree::Int)
    images = [[Int(x^g) for x in 1:degree] for g in group_elements]
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
                rows_x = block.column_rows[gx]
                vals_x = block.column_values[gx]
                rows_y = block.column_rows[gy]
                vals_y = block.column_values[gy]
                for ix in eachindex(rows_x), iy in eachindex(rows_y)
                    coefficient = inverse_order * vals_x[ix] * vals_y[iy]
                    abs(coefficient) <= 1e-14 && continue
                    term = (block_index, rows_x[ix], rows_y[iy])
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

function _tuple_sign_product(code::Int, signs::Vector{Int8}, q::Int, t::Int)
    product = 1
    for _ in 1:t
        product *= signs[(code % q) + 1]
        code ÷= q
    end
    return product
end

"""
    triangle_cb_degree_feasibility(nv, t; epsilon=1/3, solver=:mosek, verbose=true)

Build and solve SDP (18) for the triangle function on `nv` vertices and a
fixed tensor order `t`.  The split is always `s=fld(t,2)`, which minimizes the
unreduced matrix size `(m+1)^s + (m+1)^(t-s)`, where `m=binomial(nv,2)`.

Graph edge coordinates use `+1` for a present edge and `-1` for an absent
edge; the homogenizing coordinate is `+1`.  The target is `+1` for graphs
containing a triangle and `-1` otherwise.  Both sides of the absolute-value
approximation constraint in (17)--(18) are imposed.
"""
function triangle_cb_degree_feasibility(
    nv::Int,
    t::Int;
    epsilon::Real = 1 / 3,
    solver::Symbol = :mosek,
    verbose::Bool = true,
    time_limit_sec = nothing,
    reversal_symmetry::Bool = true,
)
    nv >= 3 || throw(ArgumentError("triangle finding requires nv >= 3"))
    t >= 0 || throw(ArgumentError("t must be nonnegative"))
    0 <= epsilon < 1 / 2 || throw(ArgumentError("epsilon must satisfy 0 <= epsilon < 1/2"))

    f, E, edge_index = triangle_function(nv)
    m = length(E)
    q = m + 1
    s = fld(t, 2)
    r = t - s
    use_reversal = reversal_symmetry && iseven(t)
    label_generators = _vertex_label_generators(nv, E, edge_index)
    matrix_generators, left_count, right_count = _tuple_actions(
        label_generators,
        q,
        s,
        r;
        reversal_symmetry = use_reversal,
    )
    matrix_size = left_count + right_count
    tensor_generators, tensor_size = _full_tuple_actions(
        label_generators,
        q,
        t;
        reversal_symmetry = use_reversal,
    )

    verbose && println("triangle cb-degree SDP: nv=$nv, edges=$m, t=$t, s=$s")
    verbose && println("  matrix before symmetry reduction: $matrix_size = $left_count + $right_count")
    verbose && println("  tensor coefficients before symmetry reduction: $tensor_size")
    verbose && use_reversal && println("  using tensor-reversal symmetry")

    tensor_orbit_of, tensor_orbits = _orbits_from_permutations(tensor_size, tensor_generators)
    graph_reps = _graph_orbit_representatives(nv, f, E, edge_index)
    verbose && println("  tensor coefficient orbits: $(length(tensor_orbits))")
    verbose && println("  graph/input orbits: $(length(graph_reps)) / $(length(f)) inputs")

    matrix_group = PermGroup([matrix_generators; _permutation(collect(1:matrix_size))])
    matrix_group_elements = collect(matrix_group)
    verbose && println("  induced symmetry group order: $(length(matrix_group_elements))")
    raw_blocks = _wedderburn_blocks(matrix_group, matrix_size; verbose = verbose)

    components, component_of_pair =
        _as_entry_components(q, s, r, left_count, right_count)
    component_reps =
        _component_orbit_representatives(components, component_of_pair, matrix_generators)
    verbose && println("  A_s entry components: $(length(components)), symmetry orbits: $(length(component_reps))")

    optimizer = _make_cb_optimizer(solver)
    model = solver == :mosek ? direct_model(optimizer()) : Model(optimizer)
    set_silent(model)
    solver == :mosek && set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
    time_limit_sec !== nothing && set_time_limit_sec(model, Float64(time_limit_sec))
    @objective(model, Min, 0.0)

    @variable(model, tensor[1:length(tensor_orbits)], base_name = "T")
    blocks = _block_variables(model, raw_blocks, matrix_size)
    W, entry_cache = _invariant_psd_entry_function(blocks, matrix_group_elements, matrix_size)

    # <e,lambda> <= 1.  Since A_s^*(y) and C_s(T) have zero diagonal,
    # lambda is exactly diag(W) and can be eliminated.
    diagonal_expression = AffExpr(0.0)
    _, diagonal_orbits = _orbits_from_permutations(matrix_size, matrix_generators)
    for orbit in diagonal_orbits
        diagonal_expression += length(orbit) * W(first(orbit), first(orbit))
    end
    @constraint(model, diagonal_expression <= 1.0)

    # Cross block: W + C_s(T) has no support in range(A_s^*) or Diag(lambda).
    # C_s(T) has entry T_ab/2, as in equation (11).  One representative per
    # tensor orbit suffices because both W and T have been averaged over S_nv.
    for (orbit, members) in enumerate(tensor_orbits)
        code = first(members) - 1
        left_code = code % left_count
        right_code = code ÷ left_count
        @constraint(model, W(left_code + 1, left_count + right_code + 1) + 0.5 * tensor[orbit] == 0.0)
    end

    # Within each diagonal block, W lies in range(A_s^*): its entries sum to
    # zero on every equality component.  Components in the same S_nv orbit
    # give identical constraints for invariant W.
    for component in component_reps
        expression = AffExpr(0.0)
        for (x, y) in components[component]
            expression += W(x, y)
        end
        @constraint(model, expression == 0.0)
    end

    # Approximation constraints.  Invariant T makes these constant on graph
    # isomorphism orbits, so only an unlabeled graph representative is needed.
    approximation_radius = 2.0 * Float64(epsilon)
    for graph in graph_reps
        bits = graph - 1
        signs = Vector{Int8}(undef, q)
        for edge in 1:m
            signs[edge] = ((bits >> (edge - 1)) & 1) == 1 ? Int8(1) : Int8(-1)
        end
        signs[q] = 1
        coefficients = zeros(Int, length(tensor_orbits))
        for code in 0:(tensor_size - 1)
            coefficients[tensor_orbit_of[code + 1]] += _tuple_sign_product(code, signs, q, t)
        end
        polynomial = AffExpr(0.0)
        for orbit in eachindex(coefficients)
            coefficients[orbit] == 0 && continue
            add_to_expression!(polynomial, coefficients[orbit], tensor[orbit])
        end
        target = f[graph] == 1 ? 1.0 : -1.0
        @constraint(model, polynomial - target <= approximation_radius)
        @constraint(model, polynomial - target >= -approximation_radius)
    end

    n_tensor_orbits = length(tensor_orbits)
    n_graph_orbits = length(graph_reps)
    n_as_components = length(components)
    n_as_component_orbits = length(component_reps)
    block_details = [
        (block.index, block.size, block.irrep_degree, block.multiplicity) for block in raw_blocks
    ]
    block_sizes = [(block.index, block.size) for block in raw_blocks]
    cached_matrix_entry_orbits = length(entry_cache)

    verbose && println("  reduced PSD sizes: $([size for (_, size) in block_sizes])")
    verbose && println("  affine variables: $(num_variables(model)), constraints: $(num_constraints(model; count_variable_in_set_constraints = true))")

    # The orbit construction is deliberately explicit, but its large tables
    # are not needed by the conic solver.  In particular, n=5, t=6 has over
    # 1.6 million A_s entry components.  Drop all construction-only objects
    # before the solver allocates its factorization workspace.
    tensor_orbit_of = nothing
    tensor_orbits = nothing
    tensor_generators = nothing
    graph_reps = nothing
    components = nothing
    component_of_pair = nothing
    component_reps = nothing
    diagonal_orbits = nothing
    raw_blocks = nothing
    blocks = nothing
    W = nothing
    entry_cache = nothing
    matrix_group_elements = nothing
    matrix_group = nothing
    matrix_generators = nothing
    label_generators = nothing
    GC.gc()
    try
        ccall(:malloc_trim, Cint, (Csize_t,), 0)
    catch
    end

    verbose && println("  Optimizing feasibility SDP with $solver...")
    optimize!(model)
    status = termination_status(model)
    feasible = status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL) && has_values(model)
    verbose && println("  status: $status")
    verbose && println("  feasible: $feasible")

    return (
        feasible = feasible,
        status = status,
        nv = nv,
        edges = m,
        t = t,
        s = s,
        epsilon = Float64(epsilon),
        matrix_size = matrix_size,
        tensor_size = tensor_size,
        reversal_symmetry = use_reversal,
        n_tensor_orbits = n_tensor_orbits,
        n_graph_orbits = n_graph_orbits,
        n_as_components = n_as_components,
        n_as_component_orbits = n_as_component_orbits,
        blocks = block_sizes,
        block_details = block_details,
        cached_matrix_entry_orbits = cached_matrix_entry_orbits,
        model = model,
    )
end

"""
    triangle_cb_degree(nv; t_min=0, t_max, kwargs...)

Solve the feasibility SDP successively and return the first feasible tensor
order.  Equation (18) is a family of SDPs because its dimensions depend on t.
The corresponding query complexity is `ceil(cb_degree/2)` (Theorem 4.1).
"""
function triangle_cb_degree(
    nv::Int;
    t_min::Int = 0,
    t_max::Int,
    keep_models::Bool = false,
    kwargs...,
)
    0 <= t_min <= t_max || throw(ArgumentError("require 0 <= t_min <= t_max"))
    attempts = NamedTuple[]
    for t in t_min:t_max
        result = triangle_cb_degree_feasibility(nv, t; kwargs...)
        push!(attempts, keep_models ? result : merge(result, (model = nothing,)))
        if result.feasible
            return (
                cb_degree = t,
                query_complexity = cld(t, 2),
                feasible = true,
                attempts = attempts,
            )
        end
        result = nothing
        GC.gc()
    end
    return (
        cb_degree = nothing,
        query_complexity = nothing,
        feasible = false,
        attempts = attempts,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    usage =
        "Usage:\n" *
        "  julia --project=. triangle_cb_degree_sdp.jl <nv> <t> [epsilon=1/3] [solver=mosek]\n" *
        "  julia --project=. triangle_cb_degree_sdp.jl search <nv> <t_max> [epsilon=1/3] [solver=mosek] [t_min=0]"

    isempty(ARGS) && error(usage)
    if ARGS[1] == "search"
        3 <= length(ARGS) <= 6 || error(usage)
        nv = parse(Int, ARGS[2])
        t_max = parse(Int, ARGS[3])
        epsilon = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1 / 3
        solver = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :mosek
        t_min = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 0

        println("Searching t=$t_min:$t_max for the first feasible triangle cb-degree SDP...")
        result = triangle_cb_degree(
            nv;
            t_min = t_min,
            t_max = t_max,
            epsilon = epsilon,
            solver = solver,
            verbose = true,
        )
        if result.feasible
            println("First feasible tensor order: t=$(result.cb_degree)")
            println("Quantum query complexity: $(result.query_complexity)")
        else
            println("No feasible tensor order found in t=$t_min:$t_max.")
        end
    else
        2 <= length(ARGS) <= 4 || error(usage)
        nv = parse(Int, ARGS[1])
        t = parse(Int, ARGS[2])
        epsilon = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1 / 3
        solver = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :mosek
        result = triangle_cb_degree_feasibility(nv, t; epsilon = epsilon, solver = solver)
        println(result)
    end
end
