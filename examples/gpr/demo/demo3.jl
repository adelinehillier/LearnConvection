
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
]
##
# problem  = Sequential("T")
# problem  = Sequential("dT")
# problem  = Residual("KPP", KPP.Parameters())
# problem  = Residual("TKE", TKEMassFlux.TKEParameters())
# problem  = Sequential("TKE", TKEMassFlux.TKEParameters())
problem  = Sequential("KPP", KPP.Parameters())

k = 2
logγ = -0.4
distance = antiderivative_distance
kernel   = get_kernel(k, logγ, 0.0, distance)

𝒟_train     = LearnConvection.Data.data(train, problem; D=D, N=N);
𝒟_test      = LearnConvection.Data.data(test, problem; D=D, N=N);

𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel = kernel)

anim = animate_profile_and_model_output(𝒢, 𝒟_test)
# gif(anim, "animated_profile_and_model_output.gif")
gif(anim, "all.gif"; fps = 10)

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
