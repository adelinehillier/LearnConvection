using LearnConvection
problem = Sequential("KPP"; parameters=KPP.Parameters())


train = ["general_strat_32_profiles.jld2"]

k = 2
logγ = -1.7
distance = euclidean_distance
kernel   = get_kernel(k, logγ, 0.0, distance)

𝒟 = LearnConvection.Data.data(train, problem; D=16, N=4);
𝒢 = LearnConvection.GaussianProcess.model(𝒟; kernel = kernel)

using BenchmarkTools
@btime LearnConvection.GaussianProcess.model_output(𝒟.x[200], 𝒢)
@btime 𝒢.kernel(𝒟.x[200], 𝒢.x_train[200])

@btime 𝒢.kernel(𝒢.x_train[2], 𝒢.x_train[1])

@btime predict(𝒢, 𝒟; postprocessed=true)

get_me_true_check(𝒢, 𝒟)
get_me_true_check(𝒢, 𝒟)

using Plots
anim = animate_profile_and_model_output(𝒢, 𝒟)
gif(anim, "all.gif")
