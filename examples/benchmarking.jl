using LearnConvection
problem = Sequential("KPP"; parameters=KPP.Parameters())

train = ["general_strat_8_profiles.jld2"]

k = 2
logγ = -0.4
distance = euclidean_distance
kernel   = get_kernel(k, logγ, 0.0, distance)

𝒟 = LearnConvection.Data.data(train, problem; D=16, N=4);
𝒢 = LearnConvection.GaussianProcess.model(𝒟; kernel = kernel)

using BenchmarkTools
@btime predict(𝒢, 𝒟; postprocessed=true)

@btime animate_profile(𝒢, 𝒟)
