
using LearnConvection
using Plots

train = ["general_strat_8_profiles.jld2", "general_strat_16_profiles.jld2", "general_strat_32_profiles.jld2"]

default_modify_predictor_fn(x, 𝒟, time_index) = x

problems = [
            Slack("KPP"; parameters=KPP.Parameters(), modify_predictor_fn=f),
            Slack("TKE"; parameters=TKEMassFlux.TKEParameters(), modify_predictor_fn=f),
            Slack("KPP"; parameters=KPP.Parameters(), modify_predictor_fn=f),
            Slack("TKE"; parameters=TKEMassFlux.TKEParameters(), modify_predictor_fn=f),
]

D=64
N=4

k = 3
logγ = -0.1
distance = euclidean_distance
kernel = get_kernel(k, logγ, 0.0, distance)

ranges = [1:16, 17:32, 33:48, 49:64]

p = plot(xlabel="grid point", ylabel="abs(pred-truth)", title="Slack(TKE)")
for range in ranges
    f = partial_temp_profile(range)
    problem = Slack("TKE"; parameters=TKEMassFlux.TKEParameters(), modify_predictor_fn=partial_temp_profile(range))

    𝒟_train = LearnConvection.Data.data(train, problem; D=D, N=N);
    ℳ = LearnConvection.GaussianProcess.model(𝒟_train; kernel=kernel)

    a = disparity_vector(ℳ, 𝒟_train)
    plot!(a, label="$(range)", yscale=:log10)
end
p

png(p, "abs_logyscale_Slack_TKE_kernel_$(k)_gamma_$(logγ).png");

###


p = plot(xlabel="grid point", ylabel="(pred-truth)^2", title="Slack(TKE)")
for range in ranges
    f = partial_temp_profile(range)
    problem = Slack("TKE"; parameters=TKEMassFlux.TKEParameters(), modify_predictor_fn=partial_temp_profile(range))

    𝒟_train = LearnConvection.Data.data(train, problem; D=D, N=N);
    ℳ = LearnConvection.GaussianProcess.model(𝒟_train; kernel=kernel)

    a = disparity_vector(ℳ, 𝒟_train)
    plot!(a.^2, label="$(range)", yscale=:log10)
end
p

png(p, "squared_logyscale_Slack_TKE_kernel_$(k)_gamma_$(logγ).png");
