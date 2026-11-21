# triangle_adversary_dual_restricted.jl
# Dual adversary SDP on an S_n-invariant restricted input domain.

using LinearAlgebra
using SparseArrays
using JuMP
using Clarabel
using SCS
using COSMO
import LowRankOpt as LRO
import SDPLRPlus
using PermutationGroups
using SymbolicWedderburn

const _MOSEKTOOLS_LOAD_ERROR_RESTRICTED = Ref{Any}(nothing)
try
    @eval import MosekTools
catch err
    _MOSEKTOOLS_LOAD_ERROR_RESTRICTED[] = err
end

include(joinpath(@__DIR__, "triangle_common.jl"))

function _make_restricted_optimizer(solver::Symbol)
    if solver == :clarabel
        return Clarabel.Optimizer
    elseif solver == :scs
        return SCS.Optimizer
    elseif solver == :cosmo || solver == :cosmo_scaled
        return COSMO.Optimizer
    elseif solver == :sdplrplus
        return LRO.Optimizer
    elseif solver == :mosek
        if isdefined(Main, :MosekTools)
            return getproperty(Main.MosekTools, :Optimizer)
        end
        error("solver=:mosek requested, but MosekTools could not be loaded: $(_MOSEKTOOLS_LOAD_ERROR_RESTRICTED[])")
    else
        error("Unknown solver $(solver). Use :clarabel, :scs, :cosmo, :cosmo_scaled, :sdplrplus, or :mosek.")
    end
end

function graph_orbits_only(N::Int, Sn_gens)
    parent = collect(1:N)
    function find_root(x::Int)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function unite!(a::Int, b::Int)
        ra = find_root(a)
        rb = find_root(b)
        if ra != rb
            parent[ra] = rb
        end
        return nothing
    end
    for g in Sn_gens, x in 1:N
        unite!(x, Int(x^g))
    end
    tmp = Dict{Int, Vector{Int}}()
    for x in 1:N
        push!(get!(tmp, find_root(x), Int[]), x)
    end
    orbits = Dict{Int, Vector{Int}}()
    for members in values(tmp)
        sort!(members)
        orbits[members[1]] = members
    end
    return orbits
end

function triangle_count_of_graph(nv::Int, edge_index, graph_index::Integer)
    bits = graph_index - 1
    count = 0
    for a in 1:nv, b in (a + 1):nv, c in (b + 1):nv
        one = graph_index isa BigInt ? big(1) : 1
        mask = (one << (edge_index[(a, b)] - 1)) |
               (one << (edge_index[(a, c)] - 1)) |
               (one << (edge_index[(b, c)] - 1))
        if (bits & mask) == mask
            count += 1
        end
    end
    return count
end

function graph_has_triangle(nv::Int, edge_index, graph_index::Integer)
    return triangle_count_of_graph(nv, edge_index, graph_index) > 0
end

function apply_vertex_perm_to_graph_index(
    sigma::Vector{Int},
    graph_index::Integer,
    E,
    edge_index,
)
    bits = graph_index - 1
    y = zero(graph_index - 1)
    for (k, (u, v)) in enumerate(E)
        if ((bits >> (k - 1)) & 1) == 1
            a, b = sigma[u], sigma[v]
            a > b && ((a, b) = (b, a))
            one = graph_index isa BigInt ? big(1) : 1
            y |= one << (edge_index[(a, b)] - 1)
        end
    end
    return y + 1
end

function explicit_orbit_domain(nv::Int, orbit_reps::AbstractVector{<:Integer}; verbose::Bool = true)
    E = edges_of_Kn(nv)
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    m = length(E)
    vertex_gens = Vector{Int}[]
    for i in 1:(nv - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        push!(vertex_gens, sigma)
    end

    selected = Set{typeof(first(orbit_reps))}()
    for rep in orbit_reps
        stack = [rep]
        push!(selected, rep)
        while !isempty(stack)
            x = pop!(stack)
            for sigma in vertex_gens
                y = apply_vertex_perm_to_graph_index(sigma, x, E, edge_index)
                if !(y in selected)
                    push!(selected, y)
                    push!(stack, y)
                end
            end
        end
    end

    selected_full = sort!(collect(selected))
    f = Int8[graph_has_triangle(nv, edge_index, x) ? 1 : 0 for x in selected_full]
    verbose && println("  explicit graph orbit representatives: $(sort(collect(orbit_reps)))")
    verbose && println("  explicit selected inputs: $(length(selected_full))")
    verbose && println("  selected f=0: $(count(==(Int8(0)), f)), f=1: $(count(==(Int8(1)), f))")
    return selected_full, f, E, edge_index
end

function restricted_perm_from_vertex_perm_direct(
    sigma::Vector{Int},
    selected_full::AbstractVector{<:Integer},
    full_to_local::Dict{<:Integer, Int},
    E,
    edge_index,
)
    image = Vector{Int}(undef, length(selected_full))
    for (local_x, full_x) in enumerate(selected_full)
        image[local_x] = full_to_local[
            apply_vertex_perm_to_graph_index(sigma, full_x, E, edge_index)
        ]
    end
    return graph_perm(image)
end

function restricted_full_Sn_generators_direct(nv, selected_full, full_to_local, E, edge_index)
    gens = Perm[]
    for i in 1:(nv - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        push!(gens, restricted_perm_from_vertex_perm_direct(
            sigma, selected_full, full_to_local, E, edge_index
        ))
    end
    return gens
end

function restricted_stab_e0_generators_direct(nv, selected_full, full_to_local, E, edge_index)
    gens = Perm[]
    sigma = collect(1:nv)
    sigma[1], sigma[2] = sigma[2], sigma[1]
    push!(gens, restricted_perm_from_vertex_perm_direct(
        sigma, selected_full, full_to_local, E, edge_index
    ))
    for i in 3:(nv - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        push!(gens, restricted_perm_from_vertex_perm_direct(
            sigma, selected_full, full_to_local, E, edge_index
        ))
    end
    return gens
end

function restricted_edge_transport_inverses_direct(nv, selected_full, full_to_local, E, edge_index)
    localN = length(selected_full)
    sigma_j_inv = Vector{Vector{Int}}(undef, length(E))
    for j in eachindex(E)
        u, v = E[j]
        sigma = collect(1:nv)
        pos_u = findfirst(==(u), sigma)
        sigma[pos_u], sigma[1] = sigma[1], sigma[pos_u]
        pos_v = findfirst(==(v), sigma)
        sigma[pos_v], sigma[2] = sigma[2], sigma[pos_v]
        p = restricted_perm_from_vertex_perm_direct(
            sigma, selected_full, full_to_local, E, edge_index
        )
        inv_p = zeros(Int, localN)
        for x in 1:localN
            inv_p[Int(x^p)] = x
        end
        sigma_j_inv[j] = inv_p
    end
    return sigma_j_inv
end

function restricted_perm_from_vertex_perm(
    sigma::Vector{Int},
    selected_full::Vector{Int},
    full_to_local::Dict{Int, Int},
    E,
    edge_index,
    m::Int,
    fullN::Int,
)
    p_full = vertex_perm_to_graph_perm(sigma, E, edge_index, m, fullN)
    image = Vector{Int}(undef, length(selected_full))
    for (local_x, full_x) in enumerate(selected_full)
        image[local_x] = full_to_local[Int(full_x^p_full)]
    end
    return graph_perm(image)
end

function restricted_full_Sn_generators(nv, selected_full, full_to_local, E, edge_index, m, fullN)
    gens = Perm[]
    for i in 1:(nv - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        push!(gens, restricted_perm_from_vertex_perm(
            sigma, selected_full, full_to_local, E, edge_index, m, fullN
        ))
    end
    return gens
end

function restricted_stab_e0_generators(nv, selected_full, full_to_local, E, edge_index, m, fullN)
    gens = Perm[]
    sigma = collect(1:nv)
    sigma[1], sigma[2] = sigma[2], sigma[1]
    push!(gens, restricted_perm_from_vertex_perm(
        sigma, selected_full, full_to_local, E, edge_index, m, fullN
    ))
    for i in 3:(nv - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        push!(gens, restricted_perm_from_vertex_perm(
            sigma, selected_full, full_to_local, E, edge_index, m, fullN
        ))
    end
    return gens
end

function restricted_edge_transport_inverses(nv, selected_full, full_to_local, E, edge_index, m, fullN)
    localN = length(selected_full)
    sigma_j_inv = Vector{Vector{Int}}(undef, m)
    for j in 1:m
        u, v = E[j]
        sigma = collect(1:nv)
        pos_u = findfirst(==(u), sigma)
        sigma[pos_u], sigma[1] = sigma[1], sigma[pos_u]
        pos_v = findfirst(==(v), sigma)
        sigma[pos_v], sigma[2] = sigma[2], sigma[pos_v]
        p = restricted_perm_from_vertex_perm(
            sigma, selected_full, full_to_local, E, edge_index, m, fullN
        )
        inv_p = zeros(Int, localN)
        for x in 1:localN
            inv_p[Int(x^p)] = x
        end
        sigma_j_inv[j] = inv_p
    end
    return sigma_j_inv
end

"""
    select_boundary_domain(nv; neg_min_edges, pos_max_edges, max_triangles)

Return a union of S_n graph orbits. The default keeps dense triangle-free
negative inputs and positive inputs close to the triangle boundary.
"""
function select_boundary_domain(
    nv::Int;
    neg_min_edges::Union{Nothing, Int} = nothing,
    pos_min_edges::Int = 3,
    pos_max_edges::Union{Nothing, Int} = nothing,
    max_triangles::Int = 2,
    extra_orbit_reps::AbstractVector{Int} = Int[],
    excluded_orbit_reps::AbstractVector{Int} = Int[],
    verbose::Bool = true,
)
    f, E, edge_index = triangle_function(nv)
    m = length(E)
    fullN = 1 << m
    turan_edges = fld(nv * nv, 4)
    neg_min = neg_min_edges === nothing ? max(0, turan_edges - 1) : neg_min_edges
    pos_max = pos_max_edges === nothing ? turan_edges + 1 : pos_max_edges

    Sn_gens = full_Sn_generators(nv, E, edge_index, m, fullN)
    graph_orbits = graph_orbits_only(fullN, Sn_gens)

    selected_reps = Int[]
    for rep in sort(collect(keys(graph_orbits)))
        ecount = count_ones(rep - 1)
        tcount = triangle_count_of_graph(nv, edge_index, rep)
        keep = if f[rep] == 0
            ecount >= neg_min
        else
            pos_min_edges <= ecount <= pos_max && tcount <= max_triangles
        end
        (keep || rep in extra_orbit_reps) && !(rep in excluded_orbit_reps) && push!(selected_reps, rep)
    end

    selected_full = sort!(reduce(vcat, (graph_orbits[rep] for rep in selected_reps); init = Int[]))
    f_selected = f[selected_full]
    verbose && println("  selected graph orbits: $(length(selected_reps)) / $(length(graph_orbits))")
    verbose && println("  selected inputs: $(length(selected_full)) / $fullN")
    verbose && println("  selected f=0: $(count(==(Int8(0)), f_selected)), f=1: $(count(==(Int8(1)), f_selected))")
    verbose && println("  boundary rules: neg_edges >= $neg_min, pos_edges in [$pos_min_edges,$pos_max], triangles <= $max_triangles")
    verbose && !isempty(extra_orbit_reps) && println("  explicitly included orbit representatives: $extra_orbit_reps")
    verbose && !isempty(excluded_orbit_reps) && println("  explicitly excluded orbit representatives: $excluded_orbit_reps")
    return selected_full, f, E, edge_index
end

function triangle_adversary_dual_restricted(
    nv::Int;
    solver::Symbol = :mosek,
    verbose::Bool = true,
    neg_min_edges::Union{Nothing, Int} = nothing,
    pos_min_edges::Int = 3,
    pos_max_edges::Union{Nothing, Int} = nothing,
    max_triangles::Int = 2,
    extra_orbit_reps::AbstractVector{Int} = Int[],
    excluded_orbit_reps::AbstractVector{Int} = Int[],
    explicit_orbit_reps::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    explicit_selected_full::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    collect_pair_diagnostics::Bool = false,
    time_limit_sec = nothing,
)
    selected_full, f, E, edge_index = if explicit_selected_full !== nothing
        E = edges_of_Kn(nv)
        edge_index = Dict(e => k for (k, e) in enumerate(E))
        selected = sort!(unique!(collect(explicit_selected_full)))
        f = Int8[graph_has_triangle(nv, edge_index, x) ? 1 : 0 for x in selected]
        verbose && println("  explicit selected inputs: $(length(selected))")
        verbose && println("  selected f=0: $(count(==(Int8(0)), f)), f=1: $(count(==(Int8(1)), f))")
        selected, f, E, edge_index
    elseif explicit_orbit_reps === nothing
        selected, f_full, edges, idx = select_boundary_domain(
            nv;
            neg_min_edges = neg_min_edges,
            pos_min_edges = pos_min_edges,
            pos_max_edges = pos_max_edges,
            max_triangles = max_triangles,
            extra_orbit_reps = extra_orbit_reps,
            excluded_orbit_reps = excluded_orbit_reps,
            verbose = verbose,
        )
        selected, f_full[selected], edges, idx
    else
        explicit_orbit_domain(nv, explicit_orbit_reps; verbose = verbose)
    end
    m = length(E)
    fullN = (explicit_orbit_reps === nothing && explicit_selected_full === nothing) ? (1 << m) : (big(1) << m)
    localN = length(selected_full)
    full_to_local = Dict(full_x => i for (i, full_x) in enumerate(selected_full))

    verbose && println("restricted nv=$nv: m=$m, localN=$localN")

    direct_domain = explicit_orbit_reps !== nothing || explicit_selected_full !== nothing
    Sn_gens = !direct_domain ?
        restricted_full_Sn_generators(nv, selected_full, full_to_local, E, edge_index, m, fullN) :
        restricted_full_Sn_generators_direct(nv, selected_full, full_to_local, E, edge_index)
    H_gens = !direct_domain ?
        restricted_stab_e0_generators(nv, selected_full, full_to_local, E, edge_index, m, fullN) :
        restricted_stab_e0_generators_direct(nv, selected_full, full_to_local, E, edge_index)
    H = PermGroup([H_gens; graph_perm(1:localN)])
    H_elements = collect(H)
    verbose && println("  |H| = ", length(H_elements))

    verbose && println("  Computing Wedderburn decomposition...")
    t0 = time()
    sa_basis = SymbolicWedderburn.symmetry_adapted_basis(Float64, H; semisimple = false)
    verbose && println("  done in $(round(time() - t0, digits=2))s")

    D = isempty(sa_basis) ? 0 : size(sa_basis[1].basis, 2)
    n_extra = localN - D
    raw_blocks = []
    for i in eachindex(sa_basis)
        sz = size(sa_basis[i].basis, 1)
        if sz > 0
            rows, cols, vals = findnz(sparse(sa_basis[i].basis))
            U = sparse(rows, cols, vals, sz, localN)
            push!(raw_blocks, (
                idx = i,
                U = U,
                size = sz,
                irrep_degree = SymbolicWedderburn.degree(sa_basis[i]),
                multiplicity = SymbolicWedderburn.multiplicity(sa_basis[i]),
            ))
        end
    end
    if isempty(raw_blocks) && n_extra > 0
        push!(raw_blocks, (
            idx = 1,
            U = sparse(1.0I, localN, localN),
            size = localN,
            irrep_degree = 1,
            multiplicity = localN,
        ))
        n_extra = 0
    end
    if n_extra > 0
        trivial_idx = findfirst(blk -> any(abs(blk.U[a, 1]) > 1e-10 for a in 1:blk.size), raw_blocks)
        if trivial_idx === nothing
            trivial_idx = argmax(blk.irrep_degree == 1 ? blk.size : 0 for blk in raw_blocks)
        end
        old = raw_blocks[trivial_idx]
        rows, cols, vals = findnz(old.U)
        for k in 1:n_extra
            push!(rows, old.size + k)
            push!(cols, D + k)
            push!(vals, 1.0)
        end
        raw_blocks[trivial_idx] = (
            idx = old.idx,
            U = sparse(rows, cols, vals, old.size + n_extra, localN),
            size = old.size + n_extra,
            irrep_degree = old.irrep_degree,
            multiplicity = old.multiplicity + n_extra,
        )
    end
    if solver == :sdplrplus && length(raw_blocks) > 1
        total_size = sum(blk.size for blk in raw_blocks)
        U_combined = spzeros(Float64, total_size, localN)
        offset = 0
        for blk in raw_blocks
            rows, cols, vals = findnz(blk.U)
            for k in eachindex(vals)
                U_combined[offset + rows[k], cols[k]] = vals[k]
            end
            offset += blk.size
        end
        raw_blocks = [(
            idx = 1,
            U = U_combined,
            size = total_size,
            irrep_degree = 1,
            multiplicity = total_size,
        )]
        verbose && println("  merged Wedderburn blocks into one PSD block for SDPLRPlus: size=$total_size")
    end
    block_info = [(blk.idx, blk.size, blk.irrep_degree, blk.multiplicity) for blk in raw_blocks]
    if verbose
        for (idx, sz, deg, mult) in block_info
            println("    block $idx: PSD size=$sz, irrep_degree=$deg, multiplicity=$mult")
        end
        if solver == :sdplrplus
            println("    merged PSD coordinates: $(sum(sz for (_, sz, _, _) in block_info))")
        else
            println("    represented dimensions: $(sum(deg * mult for (_, _, deg, mult) in block_info)) (should be $localN)")
        end
    end

    verbose && println("  Computing restricted graph and crossing-pair orbits...")
    t0 = time()
    graph_orbits, pair_orbits = compute_Sn_orbits(f, localN, Sn_gens; store_pair_members = false)
    graph_reps = sort(collect(keys(graph_orbits)))
    n_gorb = length(graph_reps)
    n_porb = length(pair_orbits)
    verbose && println("  done in $(round(time() - t0, digits=2))s")
    verbose && println("    graph orbits: $n_gorb")
    verbose && println("    crossing-pair orbits: $n_porb")

    local_to_graph_orbit = zeros(Int, localN)
    orbit_metadata = NamedTuple[]
    for (gidx, rep) in enumerate(graph_reps)
        members = graph_orbits[rep]
        for x in members
            local_to_graph_orbit[x] = gidx
        end
        full_rep = minimum(selected_full[x] for x in members)
        degrees = zeros(Int, nv)
        for (j, (u, v)) in enumerate(E)
            if (((full_rep - 1) >> (j - 1)) & 1) == 1
                degrees[u] += 1
                degrees[v] += 1
            end
        end
        push!(orbit_metadata, (
            local_rep = rep,
            full_rep = full_rep,
            output = Int(f[rep]),
            orbit_size = length(members),
            automorphisms = factorial(big(nv)) ÷ length(members),
            edges = count_ones(full_rep - 1),
            triangles = triangle_count_of_graph(nv, edge_index, full_rep),
            degrees = sort!(degrees; rev = true),
        ))
    end

    optimizer = _make_restricted_optimizer(solver)
    model = solver == :mosek ? direct_model(optimizer()) : Model(optimizer)
    if !verbose
        set_silent(model)
    end
    if solver == :mosek
        set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
    elseif solver == :scs
        set_optimizer_attribute(model, "max_iters", 500)
        set_optimizer_attribute(model, "eps_abs", 1e-2)
        set_optimizer_attribute(model, "eps_rel", 1e-2)
        set_optimizer_attribute(model, "verbose", verbose ? 1 : 0)
    elseif solver == :cosmo || solver == :cosmo_scaled
        set_optimizer_attribute(model, "verbose", verbose)
        set_optimizer_attribute(model, "max_iter", 10_000)
        set_optimizer_attribute(model, "eps_abs", 1e-4)
        set_optimizer_attribute(model, "eps_rel", 1e-4)
        solver == :cosmo && set_optimizer_attribute(model, "scaling", 0)
        if time_limit_sec !== nothing
            set_optimizer_attribute(model, "time_limit", Float64(time_limit_sec))
        end
    elseif solver == :sdplrplus
        set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
        set_attribute(model, "sub_solver", SDPLRPlus.Solver)
        set_attribute(model, "ranks", [8])
        set_attribute(model, "square_scalars", true)
        set_attribute(model, "printlevel", verbose ? 1 : 0)
        set_attribute(model, "printfreq", 10.0)
        set_attribute(model, "maxtime", time_limit_sec === nothing ? 1.0e12 : Float64(time_limit_sec))
        set_attribute(model, "dataset", "triangle_n$(nv)")
    end
    if time_limit_sec !== nothing && solver != :sdplrplus
        set_time_limit_sec(model, Float64(time_limit_sec))
    end

    @variable(model, t)
    @objective(model, Min, 1.0 * t)

    blocks = []
    for blk in raw_blocks
        B = if solver == :sdplrplus
            Bsym = @variable(model, [1:blk.size, 1:blk.size], Symmetric, base_name = "B_$(blk.idx)")
            @constraint(model, Bsym in PSDCone())
            Bsym
        else
            @variable(model, [1:blk.size, 1:blk.size], PSD, base_name = "B_$(blk.idx)")
        end
        col_rows = [Int[] for _ in 1:localN]
        col_vals = [Float64[] for _ in 1:localN]
        rows, cols, vals = findnz(blk.U)
        for k in eachindex(vals)
            push!(col_rows[cols[k]], rows[k])
            push!(col_vals[cols[k]], vals[k])
        end
        push!(blocks, (B = B, col_rows = col_rows, col_vals = col_vals))
    end

    H_images = [[Int(x^g) for x in 1:localN] for g in H_elements]
    inv_H_order = 1.0 / length(H_elements)

    function Xe0(x::Int, y::Int)
        terms = Dict{Tuple{Int, Int, Int}, Float64}()
        for images in H_images
            gx = images[x]
            gy = images[y]
            for (bidx, blk) in enumerate(blocks)
                rows_x = blk.col_rows[gx]
                vals_x = blk.col_vals[gx]
                rows_y = blk.col_rows[gy]
                vals_y = blk.col_vals[gy]
                for ix in eachindex(rows_x), iy in eachindex(rows_y)
                    coeff = inv_H_order * vals_x[ix] * vals_y[iy]
                    if abs(coeff) > 1e-14
                        key = (bidx, rows_x[ix], rows_y[iy])
                        terms[key] = get(terms, key, 0.0) + coeff
                    end
                end
            end
        end
        expr = AffExpr(0.0)
        for ((bidx, row, col), coeff) in terms
            add_to_expression!(expr, coeff, blocks[bidx].B[row, col])
        end
        return expr
    end

    sigma_j_inv = !direct_domain ?
        restricted_edge_transport_inverses(nv, selected_full, full_to_local, E, edge_index, m, fullN) :
        restricted_edge_transport_inverses_direct(nv, selected_full, full_to_local, E, edge_index)

    verbose && println("  Adding $n_porb crossing-pair constraints...")
    t0 = time()
    crossing_constraints = ConstraintRef[]
    crossing_endpoints = Tuple{Int, Int}[]
    crossing_local_reps = Tuple{Int, Int}[]
    crossing_hamming = Int[]
    for (k, (rep_x, rep_y, _)) in enumerate(pair_orbits)
        @assert f[rep_x] != f[rep_y] "original adversary dual only constrains opposite-output pairs"
        xbits = selected_full[rep_x] - 1
        ybits = selected_full[rep_y] - 1
        con_expr = AffExpr(0.0)
        for j in 1:m
            if ((xbits >> (j - 1)) & 1) != ((ybits >> (j - 1)) & 1)
                sx = sigma_j_inv[j][rep_x]
                sy = sigma_j_inv[j][rep_y]
                con_expr += Xe0(sx, sy)
            end
        end
        push!(crossing_constraints, @constraint(model, con_expr == 1.0))
        push!(crossing_endpoints, (
            local_to_graph_orbit[rep_x],
            local_to_graph_orbit[rep_y],
        ))
        if collect_pair_diagnostics
            push!(crossing_local_reps, (rep_x, rep_y))
            push!(crossing_hamming, count_ones(xbits ⊻ ybits))
        end
        if verbose && k % 10_000 == 0
            println("    added $k / $n_porb in $(round(time() - t0, digits=2))s")
        end
    end
    verbose && println("  done in $(round(time() - t0, digits=2))s")

    verbose && println("  Adding $n_gorb objective-bound constraints...")
    t0 = time()
    objective_constraints = ConstraintRef[]
    objective_expressions = AffExpr[]
    for rep in graph_reps
        xstar = graph_orbits[rep][1]
        obj_expr = AffExpr(0.0)
        for j in 1:m
            sx = sigma_j_inv[j][xstar]
            obj_expr += Xe0(sx, sx)
        end
        push!(objective_expressions, obj_expr)
        push!(objective_constraints, @constraint(model, obj_expr <= t))
    end
    verbose && println("  done in $(round(time() - t0, digits=2))s")

    graph_orbits = nothing
    pair_orbits = nothing
    raw_blocks = nothing
    empty!(blocks)
    empty!(H_images)
    GC.gc()
    try
        ccall(:malloc_trim, Cint, (Csize_t,), 0)
    catch
    end

    verbose && println("  Optimizing...")
    optimize!(model)
    status = termination_status(model)
    value = has_values(model) ? objective_value(model) : NaN
    diagnostics = NamedTuple[]
    pair_diagnostics = NamedTuple[]
    if has_values(model)
        objective_weights = zeros(Float64, n_gorb)
        crossing_mass = zeros(Float64, n_gorb)
        crossing_count = zeros(Int, n_gorb)
        activities = zeros(Float64, n_gorb)

        for i in eachindex(objective_constraints)
            objective_weights[i] = try
                abs(dual(objective_constraints[i]))
            catch
                0.0
            end
            activities[i] = value == 0.0 ? 0.0 : JuMP.value(objective_expressions[i]) / value
        end
        for (i, (constraint, (gx, gy))) in enumerate(zip(crossing_constraints, crossing_endpoints))
            weight = try
                abs(dual(constraint))
            catch
                0.0
            end
            crossing_mass[gx] += weight
            crossing_mass[gy] += weight
            crossing_count[gx] += 1
            crossing_count[gy] += 1
            if collect_pair_diagnostics
                rep_x, rep_y = crossing_local_reps[i]
                push!(pair_diagnostics, (
                    pair_orbit = i,
                    local_x = rep_x,
                    local_y = rep_y,
                    full_x = selected_full[rep_x],
                    full_y = selected_full[rep_y],
                    output_x = Int(f[rep_x]),
                    output_y = Int(f[rep_y]),
                    graph_orbit_x = gx,
                    graph_orbit_y = gy,
                    graph_rep_x = orbit_metadata[gx].full_rep,
                    graph_rep_y = orbit_metadata[gy].full_rep,
                    hamming = crossing_hamming[i],
                    dual_weight = weight,
                ))
            end
        end

        normalize(v) = isempty(v) || maximum(v) <= eps(Float64) ? zeros(length(v)) : v ./ maximum(v)
        objective_score = normalize(objective_weights)
        crossing_score = normalize(log1p.(crossing_mass))
        activity_score = clamp.(activities, 0.0, 1.0)

        for i in eachindex(orbit_metadata)
            meta = orbit_metadata[i]
            importance = 0.45 * objective_score[i] + 0.45 * crossing_score[i] + 0.10 * activity_score[i]
            push!(diagnostics, merge(meta, (
                objective_multiplier = objective_weights[i],
                crossing_multiplier_mass = crossing_mass[i],
                crossing_constraints = crossing_count[i],
                activity = activities[i],
                importance = importance,
                importance_per_input = importance / meta.orbit_size,
            )))
        end
    end
    verbose && println("  status: $status")
    verbose && println("  restricted ADV+/-(triangle, nv=$nv) = $value")

    return (
        value = value,
        status = status,
        n_graph_orbits = n_gorb,
        n_pair_orbits = n_porb,
        localN = localN,
        fullN = fullN,
        blocks = [(idx, sz) for (idx, sz, _, _) in block_info],
        block_details = block_info,
        selected_orbit_reps = [meta.full_rep for meta in orbit_metadata],
        orbit_diagnostics = diagnostics,
        pair_diagnostics = pair_diagnostics,
        model = model,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    nv = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
    solver = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :mosek
    max_triangles = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
    neg_min_edges = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : nothing
    pos_max_edges = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing
    result = triangle_adversary_dual_restricted(
        nv;
        solver = solver,
        max_triangles = max_triangles,
        neg_min_edges = neg_min_edges,
        pos_max_edges = pos_max_edges,
        verbose = true,
    )
    println(result)
end
