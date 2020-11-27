using LearnConvection
using Plots

D=32
N=4

## Interpolation

train = ["general_strat_4_profiles.jld2", "general_strat_32_profiles.jld2"]
test  = ["general_strat_8_profiles.jld2", "general_strat_16_profiles.jld2", "general_strat_20_profiles.jld2", "general_strat_28_profiles.jld2"]

## Extrapolation

train = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2", "general_strat_12_profiles.jld2", "general_strat_16_profiles.jld2"]
test  = ["general_strat_28_profiles.jld2", "general_strat_32_profiles.jld2"]

## Debug test

train = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2"]
test  = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2"]

## LESbrary data

train = ["free_convection", "strong_wind", "strong_wind_no_coriolis"]
test = ["weak_wind_strong_cooling", "strong_wind_weak_cooling", "strong_wind_weak_heating"]

problem  = Sequential("T")
problem  = Sequential("dT")
problem  = Residual("KPP"; parameters=KPP.Parameters())
problem  = Residual("TKE"; parameters=TKEMassFlux.TKEParameters())
problem  = Sequential("TKE"; parameters=TKEMassFlux.TKEParameters())
problem  = Sequential("KPP"; parameters=KPP.Parameters())
problem  = Sequential("dT"; modify_predictor_fn=partial_temp_profile(1:16))

default_modify_predictor_fn(x, 𝒟, time_index) = x
problem = Slack("KPP"; parameters=KPP.Parameters(), modify_predictor_fn=default_modify_predictor_fn)

problem = Sequential("TKE";
                parameters=TKEMassFlux.TKEParameters(),
                convective_adjust=false)

problem  = Sequential("dT")

k = 2
logγ = 1.0
distance = euclidean_distance
kernel   = get_kernel(k, logγ, 0.0, distance)

𝒟_train  = LearnConvection.Data.data(train, problem; D=D, N=N);
𝒟_test   = LearnConvection.Data.data(test, problem; D=D, N=N);
𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel = kernel, stencil_size = 16)
𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel = kernel)

get_me_true_check(𝒢, 𝒟_test)
# anim = animate_profile(𝒢, 𝒟_test)
anim = animate_profile(𝒢, 𝒟_train)

anim = animate_profile_and_model_output(𝒢, 𝒟_train)
gif(anim, "all.gif")

# anim = animate_profile_and_model_output(𝒢, 𝒟_test)
# gif(anim, "animated_profile_and_model_output.gif")
# gif(anim, "all.gif"; fps = 5)

p = plot_landscapes_compare_error_metrics2(k, 𝒟_train, distance, -2.0:0.1:1.0)

function plot_landscapes_compare_error_metrics2(k::Int64, 𝒟::ProfileData, distance, log_γs)
    # Compare mean log marginal likelihood with
    #    mean error on greedy check and
    #    mean error on true check

    mes  = zeros(length(log_γs)) # mean error (greedy check)
    mets  = zeros(length(log_γs)) # mean error (true check)

    for l in 1:length(log_γs)

        kernel = get_kernel(k, log_γs[l], 0.0, distance)
        ℳ = LearnConvection.GaussianProcess.model(𝒟; kernel=kernel)

        # -----compute mean error for greedy check (same as in plot log error)----
        mes[l] = get_me_greedy_check(ℳ, 𝒟)

        # -----compute mean error for true check----
        mets[l] = get_me_true_check(ℳ, 𝒟)

    end

    ylims = ( minimum([minimum(mets), minimum(mes)]) , maximum([maximum(mets), maximum(mes)]) )

    mes_plot  = plot(log_γs, mes,  xlabel="log(γ)", title="ME on greedy check, min = $(round(minimum(mes);digits=7))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([log_γs[argmin(mes)]])
    met_plot  = plot(log_γs, mets,  xlabel="log(γ)", title="ME on true check, min = $(round(minimum(mets);digits=7))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([log_γs[argmin(mets)]])

    layout = @layout [a; b]
    return plot(mes_plot, met_plot, layout = layout)
end


function animate_profile(ℳ, 𝒟)

    variable = 𝒟.problem.variable # "T" or "wT"
    xlims = (18,20)

    predi = predict(ℳ, 𝒟; postprocessed=true)

    animation_set = 1:1:12
    anim = @animate for i in animation_set
        exact = 𝒟.v[:,i]
        day_string = string(floor(Int, 𝒟.t[i]/86400))
        scatter(predi[i], 𝒟.zavg, label = "GP")
        plot!(exact, 𝒟.z, legend = :topleft, label = "LES", ylabel = "Depth [m]", title = "day " * day_string, xlims=xlims)
    end

    return anim
end
