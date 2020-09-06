"""
This example uses the ProfileData struct and GP.
"""

include("GP1.jl")

log_γs = -3.0:0.1:3.0

V_name = Dict("T" =>"Temperature [°C]", "wT"=>"Temperature flux [°C⋅m/s]")

v_str = "T"
# v_str = "wT"

N = 1
# N = 2

function animate_profile_trained_on_any(filename_predict, filenames_train, k::Int64, γ, d, D)

    𝒟predict = construct_profile_data(filename_predict, v_str, D; N=N)
    kernel = get_kernel(k, γ, 0.0)

    𝒟train = construct_profile_data_multiple(filenames_train, v_str, D; N=N)
    𝒢 = construct_gpr(𝒟train.x_train, 𝒟train.y_train, kernel; distance_fn=d, z=𝒟train.zavg);
    println(length(𝒟predict.x_train))

    println(length(𝒟train.x_train))
    gpr_prediction = get_gpr_pred(𝒢, 𝒟predict)

    animation_set = 1:10:(𝒟predict.Nt-2)
    anim = @animate for i in animation_set
        exact = 𝒟predict.v[:,i]
        day_string = string(floor(Int, 𝒟predict.t[i]/86400))
        p1 = scatter(gpr_prediction[i], 𝒟predict.zavg, label = "GP")
        # xlims=(minimum(data.v[:,1]),maximum(data.v[:,1]))
        xlims=(18,20)

        if i<size(𝒟train.v)[2]
            exact16 = 𝒟train.v[:,i]
        else
            exact16 = 𝒟train.v[:,end]
        end

        plot!(exact, 𝒟predict.z, legend = :topleft, label = "LES", xlabel = "$(V_name["T"])", ylabel = "Depth [m]", title = "day $(day_string)", xlims=xlims)
        # plot!(exact16, 𝒟train.z, legend = :topleft, label = "LES gs 16", xlabel = "$(V_name["T"])", ylabel = "depth", xlims=xlims)
        display(p1)
    end

    return anim
end

# function l2norm_strat_penalty(a,b,z) # d(x,x') = || x - x' ||
#     α_proxy(x) = x[2] - x[1]
#     # println("hello")
#     # println("$(abs(α_proxy(a)-α_proxy(b)))")
#     if abs(α_proxy(a)-α_proxy(b))>0.05
#         return l2_norm(a,b) + 0.0001
#     end
#     return l2_norm(a,b)
# end


filename="general_strat_32_profiles"
filenames = [
             "general_strat_4_profiles.jld2",
             # "general_strat_8_profiles.jld2",
             # "general_strat_12_profiles.jld2",
             # "general_strat_16_profiles.jld2",
             # "general_strat_20_profiles.jld2",
             # "general_strat_24_profiles.jld2",
             # "general_strat_28_profiles.jld2",
             # "general_strat_32_profiles.jld2"
             ]
filenames_descript = "8"
# filenames = ["general_strat_8_profiles.jld2","general_strat_32_profiles.jld2"]

γ=5.5
k=2
anim = animate_profile_trained_on_any("$(filename).jld2", filenames, k, γ, l2norm_strat_penalty, 16)
gif(anim, pwd() * "/../les/data_sandreza/$(filename)/gp_γ$(γ)_k$(k)_l2norm_strat_penalty_trainedOnGs$(filenames_descript)_scaledwithtime_N$(N).gif", fps = 15)
