using LinearAlgebra
using SparseArrays
using Printf
using SDPLRPlus

function all_inputs(n::Int)
    xs = Vector{Vector{Int}}()
    for mask in 0:(2^n - 1)
        push!(xs, [((mask >> j) & 1) for j in 0:(n - 1)])
    end
    xs
end

threshold_value(n::Int, k::Int) = sqrt(k * (n - k + 1))

function sym_entry(dim::Int, p::Int, q::Int, v::Float64)
    if p == q
        return sparse([p], [p], [v], dim, dim)
    end
    return sparse([p, q], [q, p], [v / 2, v / 2], dim, dim)
end

function coord_index(i::Int, a::Int, m::Int)
    return (i - 1) * m + a
end

function build_threshold_sdp(n::Int, k::Int; mode::Symbol, alpha::Float64=1.0)
    xs = all_inputs(n)
    ys = [sum(x) >= k ? 1 : 0 for x in xs]
    m = length(xs)
    has_T = mode == :value
    dim = n * m + (has_T ? 1 : 0)
    t_idx = dim

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    if mode == :value
        push!(rows, t_idx); push!(cols, t_idx); push!(vals, 1.0)
    elseif mode == :trace
        for i in 1:n, a in 1:m
            idx = coord_index(i, a, m)
            push!(rows, idx); push!(cols, idx); push!(vals, 0.5)
        end
    else
        error("mode must be :value or :trace")
    end
    C = sparse(rows, cols, vals, dim, dim)

    As = Any[]
    b = Float64[]
    constraint_types = Bool[]

    # Per-input cost constraints. Inequality means <A,X> <= b.
    cap = alpha * threshold_value(n, k) / 2.0 + 1e-6
    for a in 1:m
        rr = Int[]
        cc = Int[]
        vv = Float64[]
        for i in 1:n
            idx = coord_index(i, a, m)
            push!(rr, idx); push!(cc, idx); push!(vv, 0.5)
        end
        rhs = cap
        if mode == :value
            push!(rr, t_idx); push!(cc, t_idx); push!(vv, -1.0)
            rhs = 0.0
        end
        push!(As, sparse(rr, cc, vv, dim, dim))
        push!(b, rhs)
        push!(constraint_types, true)
    end

    # Adversary equalities for unordered pairs.
    for a in 1:m
        for c in a:m
            rr = Int[]
            cc = Int[]
            vv = Float64[]
            for i in 1:n
                if xs[a][i] != xs[c][i]
                    p = coord_index(i, a, m)
                    q = coord_index(i, c, m)
                    if p == q
                        push!(rr, p); push!(cc, p); push!(vv, 1.0)
                    else
                        push!(rr, p); push!(cc, q); push!(vv, 0.5)
                        push!(rr, q); push!(cc, p); push!(vv, 0.5)
                    end
                end
            end
            push!(As, sparse(rr, cc, vv, dim, dim))
            push!(b, ys[a] != ys[c] ? 1.0 : 0.0)
            push!(constraint_types, false)
        end
    end

    return C, As, b, constraint_types, m, dim
end

function block_ranks(Rt, n::Int, m::Int; tol::Float64=1e-6)
    X = Matrix(transpose(Rt) * Rt)
    ranks = Int[]
    for i in 1:n
        lo = (i - 1) * m + 1
        hi = i * m
        B = Symmetric(X[lo:hi, lo:hi])
        ev = eigvals(B)
        cutoff = max(tol, tol * maximum(abs.(ev)))
        push!(ranks, count(>(cutoff), ev))
    end
    return ranks
end

function run_case(n::Int, k::Int; mode::Symbol, rank::Int, alpha::Float64=1.0, maxtime::Float64=30.0)
    C, As, b, ctypes, m, dim = build_threshold_sdp(n, k; mode=mode, alpha=alpha)
    result = nothing
    t = @elapsed redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            result = sdplr(
                C,
                As,
                b,
                rank;
                constraint_types=ctypes,
                ptol=1e-6,
                ptol_mode=:absolute,
                objtol=Inf,
                gtol=1e-6,
                gtol_mode=:absolute,
                maxtime=maxtime,
                printlevel=0,
                printfreq=10.0,
                maxmajoriter=10_000,
            )
        end
    end
    ranks = block_ranks(result["Rt"], n, m)
    @printf(
        "mode=%s n=%d k=%d alpha=%.6g rank=%d time=%.3f primal=%.3e obj=%.8g ranks=%s\n",
        String(mode),
        n,
        k,
        alpha,
        rank,
        t,
        get(result, "primal_vio", NaN),
        get(result, "obj", NaN),
        join(ranks, ";"),
    )
    return result
end

function main()
    mode = Symbol(get(ENV, "MODE", "trace"))
    n = parse(Int, get(ENV, "N", "4"))
    k = parse(Int, get(ENV, "K", "2"))
    alpha = parse(Float64, get(ENV, "ALPHA", "1.0"))
    rank = parse(Int, get(ENV, "RANK", "4"))
    maxtime = parse(Float64, get(ENV, "MAXTIME", "30.0"))
    run_case(n, k; mode=mode, rank=rank, alpha=alpha, maxtime=maxtime)
end

main()
