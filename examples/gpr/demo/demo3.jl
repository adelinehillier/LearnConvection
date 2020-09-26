using LearnConvection
using Plots

D=16
N=4

## Interpolation

train = ["general_strat_4_profiles.jld2", "general_strat_32_profiles.jld2"]
test = ["general_strat_8_profiles.jld2", "general_strat_16_profiles.jld2", "general_strat_20_profiles.jld2", "general_strat_28_profiles.jld2"]

## Extrapolation

train = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2", "general_strat_12_profiles.jld2", "general_strat_16_profiles.jld2"]
test = ["general_strat_28_profiles.jld2", "general_strat_32_profiles.jld2"]

## Debug test

train = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2"]
test = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2"]

##

train = ["general_strat_28_profiles.jld2"]
test = train

problem  = Sequential("T")
# problem  = Sequential("dT")
# problem  = Residual("KPP", KPP.Parameters())
# problem  = Residual("TKE", TKEMassFlux.TKEParameters())
# problem  = Sequential("TKE", TKEMassFlux.TKEParameters())
# problem  = Sequential("KPP", KPP.Parameters())

# k = 2
# logγ = -0.4
# distance = antiderivative_distance
# kernel   = get_kernel(k, logγ, 0.0, distance)

𝒟_train     = LearnConvection.Data.data(train, problem; D=D, N=N);
𝒟_test      = LearnConvection.Data.data(test, problem; D=D, N=N);

𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel = kernel)

anim = animate_profile(𝒢, 𝒟_test)

# anim = animate_profile_and_model_output(𝒢, 𝒟_test)
# gif(anim, "animated_profile_and_model_output.gif")
gif(anim, "all.gif"; fps = 5)




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

## Example for documentation

using LearnConvection
using Plots

# simulation data
train    = ["general_strat_8_profiles.jld2", "general_strat_16_profiles.jld2"]
test     = "general_strat_32_profiles.jld2"
D        = 32
N        = 4

# problem
problem  = Residual("TKE", TKEMassFlux.TKEParameters())

# kernel
k = 2
logγ = -0.4
distance = antiderivative_distance
kernel   = get_kernel(k, logγ, 0.0, distance)

# data
𝒟_train  = LearnConvection.Data.data(train, problem; D=D, N=N);
𝒟_test   = LearnConvection.Data.data(test, problem; D=D, N=N);

# model 𝒢 trained on 𝒟_train
𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel = kernel)

# animate the result, where 𝒢 is tested on 𝒟_test
anim = animate_profile_and_model_output(𝒢, 𝒟_test)
gif(anim, "all.gif")
