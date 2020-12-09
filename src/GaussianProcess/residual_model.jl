"""
Residual model to pick up the slack between approximation of -∂z(wT) and the truth.
This example uses the ResidualData struct and GP.

    # temperature eq.
    # ∂t(T) + ∂x(uT) + ∂y(vT) + ∂z(wT) = 𝓀∇²T
    # --> horizontal average -->
    #
    #           diffusive  advective
    # ∂t(T) = - ∂z(wT) +  𝓀∂z(∂z(T))
    #
    # GOAL: express wT in terms of only large-scale terms wT = F(T,h,𝓀,μ)
    #  - 𝓀: diffusivity
    #  - μ: coefficient of viscosity

    # T(n+1) = T(n) + Δt * (- ∂z(wT) +  𝓀∂z(∂z(T)))
    # Gp = -∂z(wT)
    # y = T(n+1) - T(n) - Δt * ∂z(∂z(T))

    use GPR to capture the difference between the truth
        Δt(-∂z(wT))
    and the approximation
        T(n+1) - T(n) - Δt * ∂z(∂z(T))

    # target
        approx - truth
"""

# using Statistics, LinearAlgebra, Plots
# include("GP.jl")

struct ResidualData
    truth
    approx
end

# get_truth(𝒟::ProfileData, 𝒢::GP)

δ(ϕ, z) = diff(ϕ) ./ diff(z)

function construct_residual_data(filename, D; N=4)

    # return ProfileData(V, vavg, x, y, x_train, y_train, verification_set, z, zavg, t, Nt, n_train)
end


# filename = "general_strat_16_profiles.jld2"
filename = "dns_profiles.jld2"
D = 16
N = 4

𝒟_wT = construct_profile_data(filename, "wT", D; N=N)
𝒟_T  = construct_profile_data(filename, "T",  D; N=N)
Nt = 𝒟_wT.Nt
t = 𝒟_wT.t
Δt = t[2]-t[1]
κₑ = 𝒟_wT.κₑ

avgd = false
smooth = false

if avgd
    wT = 𝒟_wT.vavg
    T = 𝒟_T.vavg
    z = 𝒟_wT.zavg
    ∂wT∂z = [δ(q, z) for q in wT] # Δt(∂z(wT))
    # Δt * ∂z(∂z(T)) + T(n) - T(n+1)
    Δt∂²T∂z² = [δ( δ(q,z), z[1:end-1] )*Δt for q in T]
    approx = [κₑ*Δt∂²T∂z²[i] .+ T[i][1:end-2] .- T[i+1][1:end-2] for i in 1:(Nt-1)]
else
    wT = 𝒟_wT.v
    if smooth
        wT = smooth_window(wT)
    end

    T  = 𝒟_T.v
    z  = 𝒟_wT.z
    ∂wT∂z = [δ(wT[:,i+1], z)*Δt for i in 1:Nt-1] # Δt(∂z(wT))
    # Δt * ∂z(∂z(T)) + T(n) - T(n+1)
    Δt∂²T∂z² = [δ( δ(T[:,i],z), z[1:end-1] )*Δt for i in 1:Nt]
    approx = [κₑ.*2*Δt∂²T∂z²[i] .+ T[:,i][1:end-2] .- T[:,i+2][1:end-2] for i in 1:(Nt-2)]
end

# Δt(-∂z(wT)) = T(n+1) - T(n) - Δt * ∂z(∂z(T))
# Δt(∂z(wT)) = Δt * ∂z(∂z(T)) + T(n) - T(n+1)

# animation_set = 1:20:(Nt)
animation_set = 1:50
anim = @animate for i in animation_set
    day_string = string(floor(Int, t[i]/86400))
    p1 = scatter(∂wT∂z[i], z, label = "Δt*∂z(wT)", xlims=(-2e-6,2e-6))
    # scatter!(approx[i], z[1:end-2], legend = :topleft, label = "∂z(∂z(T)) + T(n) - T(n+1)", ylabel = "depth", title = "day " * day_string)
    scatter!(approx[i], z[1:end-2], legend = :topright, label = "Δt*κₑ*∂z(∂z(T)) + T(n) - T(n+1)", ylabel = "depth", title = "t = $(𝒟_wT.t[i])")
end
gif(anim, pwd() * "/try2_dns.gif", fps = 2)



function smooth_window(wT)
    # takes in an Nz x Nt array
    Nz, Nt = size(wT)
    smooth_wT = similar(wT)
    for i in 2:(Nz-1)
        for j in 1:Nt
            smooth_wT[i,j] = (wT[i-1,j] + wT[i+1,j])/2
        end
    end
    return smooth_wT
end

smooth_window(wT)
