# triangle_common.jl
# Shared routines for the symmetry-reduced adversary SDP for triangle finding.

using PermutationGroups

function graph_perm(image::AbstractVector{<:Integer})
    n = length(image)
    if n <= typemax(UInt16)
        return Perm{UInt16}(collect(Int, image))
    elseif n <= typemax(UInt32)
        return Perm{UInt32}(collect(Int, image))
    else
        return Perm{UInt64}(collect(Int, image))
    end
end

"""
    edges_of_Kn(nv) -> Vector{Tuple{Int,Int}}

Edges of K_nv in lexicographic order, 1-indexed.
"""
function edges_of_Kn(nv::Int)
    return [(i, j) for i in 1:nv for j in (i + 1):nv]
end

"""
    triangle_function(nv) -> (f, edges, edge_index)

`f[x+1] == 1` iff the graph encoded by the m-bit integer `x` contains a triangle.
The graph vertices are `1:nv`, and the bits are ordered by `edges_of_Kn(nv)`.
"""
function triangle_function(nv::Int)
    E = edges_of_Kn(nv)
    m = length(E)
    N = 1 << m
    edge_index = Dict(e => k for (k, e) in enumerate(E))
    triangles = [(a, b, c) for a in 1:nv for b in (a + 1):nv for c in (b + 1):nv]

    f = zeros(Int8, N)
    for x in 0:(N - 1)
        for (a, b, c) in triangles
            i_ab = edge_index[(a, b)]
            i_ac = edge_index[(a, c)]
            i_bc = edge_index[(b, c)]
            mask = (1 << (i_ab - 1)) | (1 << (i_ac - 1)) | (1 << (i_bc - 1))
            if (x & mask) == mask
                f[x + 1] = 1
                break
            end
        end
    end
    return f, E, edge_index
end

"""
    vertex_perm_to_graph_perm(sigma, E, edge_index, m, N) -> Perm

Convert a vertex permutation `sigma` into its induced permutation on the `N=2^m`
labeled graphs. Graphs are represented by indices `1:N`, i.e. bitstring `x-1`.
"""
function vertex_perm_to_graph_perm(sigma::Vector{Int}, E, edge_index, m::Int, N::Int)
    edge_perm = zeros(Int, m)
    for (k, (u, v)) in enumerate(E)
        a, b = sigma[u], sigma[v]
        if a > b
            a, b = b, a
        end
        edge_perm[k] = edge_index[(a, b)]
    end

    image = zeros(Int, N)
    for x in 0:(N - 1)
        y = 0
        for k in 1:m
            if ((x >> (k - 1)) & 1) == 1
                y |= 1 << (edge_perm[k] - 1)
            end
        end
        image[x + 1] = y + 1
    end
    return graph_perm(image)
end

"""
    stab_e0_generators(nv, E, edge_index, m, N) -> Vector{Perm}

Generators for H = Stab({1,2}) = S_2 x S_{3,...,nv}, acting on graph inputs.
"""
function stab_e0_generators(nv::Int, E, edge_index, m::Int, N::Int)
    gens = Perm[]

    # Swap vertices 1 and 2.
    sigma = collect(1:nv)
    sigma[1], sigma[2] = sigma[2], sigma[1]
    push!(gens, vertex_perm_to_graph_perm(sigma, E, edge_index, m, N))

    # Adjacent transpositions generating S_{3,...,nv}.
    for i in 3:(nv - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        push!(gens, vertex_perm_to_graph_perm(sigma, E, edge_index, m, N))
    end
    return gens
end

"""
    full_Sn_generators(nv, E, edge_index, m, N) -> Vector{Perm}

Adjacent transpositions generating the full S_nv action on graph inputs.
"""
function full_Sn_generators(nv::Int, E, edge_index, m::Int, N::Int)
    gens = Perm[]
    for i in 1:(nv - 1)
        sigma = collect(1:nv)
        sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
        push!(gens, vertex_perm_to_graph_perm(sigma, E, edge_index, m, N))
    end
    return gens
end

"""
    compute_Sn_orbits(f, N, Sn_gens; store_pair_members=false)
        -> (graph_orbits, pair_orbits)

Compute S_nv-orbits of graph inputs and crossing-pair orbits.

`graph_orbits` is `Dict{Int,Vector{Int}}`, keyed by the minimum member of each orbit.

`pair_orbits` is a vector of `(rep_zero, rep_one, members)`. It contains one
representative for each orbit of crossing ordered pairs `(x,y)` with `f[x]=0`,
`f[y]=1`. When `store_pair_members=false`, `members` is empty, which is the mode
used by the dual SDP and is the only practical mode for nv=6.
"""
function compute_Sn_orbits(f::Vector{Int8}, N::Int, Sn_gens; store_pair_members::Bool = false)
    # Graph orbits by union-find.
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

    for g in Sn_gens
        for x in 1:N
            unite!(x, Int(x^g))
        end
    end

    tmp = Dict{Int, Vector{Int}}()
    for x in 1:N
        r = find_root(x)
        push!(get!(tmp, r, Int[]), x)
    end

    # Canonicalize by minimum member for stable output.
    graph_orbits = Dict{Int, Vector{Int}}()
    for members in values(tmp)
        sort!(members)
        graph_orbits[members[1]] = members
    end

    zero_reps = sort([rep for (rep, members) in graph_orbits if f[rep] == 0])
    one_graphs = findall(==(Int8(1)), f)
    if isempty(zero_reps) || isempty(one_graphs)
        return graph_orbits, Tuple{Int, Int, Vector{Tuple{Int, Int}}}[]
    end

    identity_N = graph_perm(1:N)
    Sn = PermGroup([Sn_gens; identity_N])
    Sn_elements = collect(Sn)

    pair_orbits = Tuple{Int, Int, Vector{Tuple{Int, Int}}}[]

    for x0 in zero_reps
        stabilizer = [g for g in Sn_elements if Int(x0^g) == x0]
        seen_y = falses(N)

        for y0 in one_graphs
            seen_y[y0] && continue

            stack = [y0]
            seen_y[y0] = true
            stab_orbit = Int[]

            while !isempty(stack)
                y = pop!(stack)
                if store_pair_members
                    push!(stab_orbit, y)
                end
                for g in stabilizer
                    gy = Int(y^g)
                    if !seen_y[gy]
                        seen_y[gy] = true
                        push!(stack, gy)
                    end
                end
            end

            members = Tuple{Int, Int}[]
            if store_pair_members
                members_set = Set{Tuple{Int, Int}}()
                for y in stab_orbit, g in Sn_elements
                    gx = Int(x0^g)
                    gy = Int(y^g)
                    p = gx < gy ? (gx, gy) : (gy, gx)
                    push!(members_set, p)
                end
                members = sort!(collect(members_set))
            end

            push!(pair_orbits, (x0, y0, members))
        end
    end

    return graph_orbits, pair_orbits
end

"""
    edge_transport_inverses(nv, E, edge_index, m, N) -> Vector{Vector{Int}}

For each edge j, produce the inverse graph permutation of a vertex permutation
that maps the canonical edge {1,2} to E[j]. The dual uses
`X_j[x,y] = X_e0[sigma_j^{-1}(x), sigma_j^{-1}(y)]`.
"""
function edge_transport_inverses(nv::Int, E, edge_index, m::Int, N::Int)
    sigma_j_inv = Vector{Vector{Int}}(undef, m)
    for j in 1:m
        u, v = E[j]
        sigma = collect(1:nv)

        pos_u = findfirst(==(u), sigma)
        sigma[pos_u], sigma[1] = sigma[1], sigma[pos_u]

        pos_v = findfirst(==(v), sigma)
        sigma[pos_v], sigma[2] = sigma[2], sigma[pos_v]

        p = vertex_perm_to_graph_perm(sigma, E, edge_index, m, N)
        inv_p = zeros(Int, N)
        for x in 1:N
            inv_p[Int(x^p)] = x
        end
        sigma_j_inv[j] = inv_p
    end
    return sigma_j_inv
end
