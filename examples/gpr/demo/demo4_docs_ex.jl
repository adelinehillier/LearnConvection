## Example for documentation

using LearnConvection
using Plots

# problem
problem  = Slack("TKE"; parameters=TKEMassFlux.TKEParameters())

# data
train    = ["general_strat_32_profiles.jld2"]
test     = ["general_strat_32_profiles.jld2"]
D        = 32
N        = 4

# kernel
k        = 1
logγ     = 0.4
logσ     = 0.0
distance = euclidean_distance
kernel   = get_kernel(k, logγ, logσ, distance)

# data
𝒟_train  = LearnConvection.Data.data(train, problem; D=D, N=N);
𝒟_test   = LearnConvection.Data.data(test, problem; D=D, N=N);

# 𝒢 is trained on 𝒟_train
𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel = kernel, stencil_size=8)

# animate the mean GP prediction, where 𝒢 is tested on 𝒟_test
anim = animate_profile_and_model_output(𝒢, 𝒟_test)
gif(anim, "basic_example_8.gif", fps=15)
