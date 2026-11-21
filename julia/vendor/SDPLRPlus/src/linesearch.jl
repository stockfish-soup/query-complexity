"""
Exact line search for minimizing the augmented Lagrangian
"""
function linesearch!(
    var::SolverVars{Ti,Tv}, aux, Dt::AbstractArray{Tv}; α_max=one(Tv)
) where {Ti<:Integer,Tv}
    m = length(var.primal_vio_raw) - 1
    # evaluate 𝓐(RDᵀ + DRᵀ)
    RD_dt = @elapsed begin
        𝒜!(var.A_RD, aux, var.Rt, Dt)
    end
    # remember we divide it by 2 in Aoper, now scale back
    var.A_RD .*= 2.0
    # evaluate 𝓐(DDᵀ)
    DD_dt = @elapsed begin
        𝒜!(var.A_DD, aux, Dt, Dt)
    end
    @debug "RD_dt, DD_dt" RD_dt, DD_dt

    biquadratic = zeros(5)
    cubic = zeros(4)

    # p0 = ⟨ C, RRᵀ⟩           = var.obj[]
    # p1 = ⟨ C, (RDᵀ + DRᵀ)⟩   = C_RD
    # p2 = ⟨ C, DDᵀ⟩           = C_DD
    # (-q0) = 𝓐(RRᵀ) - b      = var.primal_vio_raw
    # q1 = 𝓐(RDᵀ + DRᵀ)       = 𝓐_RD
    # q2 = 𝓐(DDᵀ)             = 𝓐_DD

    # f(x) = a x⁴ + b x³ + c x² + d x + e
    # a = σ / 2 * ||q2||²
    # b = σ * q1ᵀ * q2
    # c = p2 - λᵀ * q2 + σ (-q0)ᵀ * q2 + σ /2 ||q1||²
    # d = p1 - λᵀ * q1 + σ (-q0)ᵀ * q1
    # e = p0 - λᵀ * (-q0) + σ / 2 * ||-q0||²
    p0 = var.obj[]
    p1 = var.A_RD[m+1]
    p2 = var.A_DD[m+1]
    neg_q0 = @view var.primal_vio_raw[1:m]
    q1 = @view var.A_RD[1:m]
    q2 = @view var.A_DD[1:m]
    σ = var.σ[]

    biquadratic[1] = (p0 - dot(var.λ, neg_q0) + σ * dot(neg_q0, neg_q0) / 2)

    # in principle biquadratic[2] should equal to
    # the inner product between direction and gradient
    # thus it should be negative

    biquadratic[2] = (p1 - dot(var.λ, q1) + σ * dot(neg_q0, q1))

    biquadratic[3] = (p2 - dot(var.λ - σ * neg_q0, q2) + σ * dot(q1, q1) / 2)

    biquadratic[4] = σ * dot(q1, q2)

    biquadratic[5] = σ * dot(q2, q2) / 2

    cubic[1] = 1.0 * biquadratic[2]

    if cubic[1] > eps()
        error("Error: cubic[1] = $(cubic[1]) should be less than 0.")
    end

    cubic[2] = 2.0 * biquadratic[3]

    cubic[3] = 3.0 * biquadratic[4]

    cubic[4] = 4.0 * biquadratic[5]

    if abs(cubic[4]) < eps()
        # got a quadractic function
        # error("Error: cubic[4] is zero, got a quadratic function")
        quadratic = cubic[1:3]
        quadratic ./= quadratic[3]
        f = Polynomial(biquadratic)
        df = Polynomial(quadratic)

        f0 = biquadratic[1] # f(α=0)
        α_star = 0.0 # optimal α
        f_star = f0 # optimal f(α)

        Roots = PolynomialRoots.roots(quadratic)
        push!(Roots, α_max)

    else
        cubic ./= cubic[4]
        f = Polynomial(biquadratic)
        df = Polynomial(cubic)

        f0 = biquadratic[1] # f(α=0)
        α_star = 0.0 # optimal α
        f_star = f0 # optimal f(α)

        Roots = PolynomialRoots.roots(cubic)
        push!(Roots, α_max)
    end

    for i in eachindex(Roots)
        # only examine real roots in [0, α_max]
        if (abs(imag(Roots[i])) >= eps())
            continue
        end
        root = real(Roots[i])
        if (root < 0) || (root > α_max)
            continue
        end
        f_α = f(root)
        if f_α < f_star
            f_star = f_α
            α_star = root
        end
    end

    # update the primal violation and function value
    # notice that
    # 𝓐((R + αD)(R + αD)ᵀ) =
    # 𝓐(RRᵀ) + α 𝓐(RDᵀ + DRᵀ) + α² 𝓐(DDᵀ)
    @. var.primal_vio_raw += α_star * (α_star * var.A_DD + var.A_RD)
    var.obj[] = var.primal_vio_raw[m+1]

    # keep primal_vio in sync: primal_vio[i] = max(primal_vio_raw[i], lb[i])
    @inbounds for i in 1:m
        var.primal_vio[i] = max(var.primal_vio_raw[i], var.primal_vio_lb[i])
    end

    return α_star, f_star
end

"""
Armijo backtracking line search for the sharp augmented Lagrangian with inequality constraints.

The exact quartic linesearch is only valid when all constraints are equalities (the AL is then
a degree-4 polynomial in α). For inequality constraints the AL involves `min(λ_ub, λ - σg)`
which creates a piecewise structure. This function evaluates the **true** sharp AL cheaply at
each candidate step using the pre-computed quadratic updates
    gᵢ(α) = primal_vio_raw[i] + α·A_RD[i] + α²·A_DD[i]
and backtracks with the Armijo sufficient-decrease condition.
"""
function linesearch_armijo!(
    var::SolverVars{Ti,Tv}, aux, Dt::AbstractArray{Tv}; α_max=one(Tv)
) where {Ti<:Integer,Tv}
    m = length(var.primal_vio_raw) - 1

    # compute 𝒜(RDᵀ + DRᵀ) and 𝒜(DDᵀ) — same first steps as linesearch!
    RD_dt = @elapsed begin
        𝒜!(var.A_RD, aux, var.Rt, Dt)
    end
    var.A_RD .*= 2
    DD_dt = @elapsed begin
        𝒜!(var.A_DD, aux, Dt, Dt)
    end
    @debug "RD_dt, DD_dt" RD_dt, DD_dt

    σ = var.σ[]

    # Evaluate the true sharp AL at step α using O(m) arithmetic (no matrix ops).
    function eval_AL(α::Tv)
        ℒ = var.obj[] + α * var.A_RD[m+1] + α^2 * var.A_DD[m+1]
        @inbounds for i in 1:m
            g_i = var.primal_vio_raw[i] + α * var.A_RD[i] + α^2 * var.A_DD[i]
            λ̃ = min(var.λ_ub[i], var.λ[i] - σ * g_i)
            ℒ += (λ̃^2 - var.λ[i]^2) / (2σ)
        end
        return ℒ
    end

    ℒ_0 = eval_AL(zero(Tv))

    # Directional derivative at α=0:  dℒ/dα|₀ = A_RD[m+1] + dot(y[1:m], A_RD[1:m])
    # var.y[i] = -min(λ_ub[i], λ[i] - σ·primal_vio_raw[i]) was set by the preceding g! call.
    slope = var.A_RD[m+1] + dot(@view(var.y[1:m]), @view(var.A_RD[1:m]))

    c = Tv(1e-4)    # standard Armijo constant
    α = α_max
    ℒ_α = eval_AL(α)

    for _ in 1:50     # at most 50 halvings (α_min ≈ 10^{-15} α_max)
        ℒ_α ≤ ℒ_0 + c * α * slope && break
        α /= 2
        ℒ_α = eval_AL(α)
    end

    # commit the accepted step: update primal_vio_raw, obj, primal_vio
    @. var.primal_vio_raw += α * (α * var.A_DD + var.A_RD)
    var.obj[] = var.primal_vio_raw[m+1]
    @inbounds for i in 1:m
        var.primal_vio[i] = max(var.primal_vio_raw[i], var.primal_vio_lb[i])
    end

    return α, ℒ_α
end
