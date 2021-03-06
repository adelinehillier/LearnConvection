
using LearnConvection
using Plots

## Extrapolation

train = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2", "general_strat_12_profiles.jld2", "general_strat_16_profiles.jld2"]
validate = ["general_strat_20_profiles.jld2", "general_strat_24_profiles.jld2"]
test = ["general_strat_28_profiles.jld2", "general_strat_32_profiles.jld2"]

Qs = [1,2]
problems = [
            Slack("KPP"; parameters=KPP.Parameters()),
            Slack("TKE"; parameters=TKEMassFlux.TKEParameters()),
            Residual("KPP"; parameters=KPP.Parameters()),
            Residual("TKE"; parameters=TKEMassFlux.TKEParameters()),
            Sequential("TKE"; parameters=TKEMassFlux.TKEParameters()),
            Sequential("KPP"; parameters=KPP.Parameters()),
            Sequential("dT")
]

for Q in Qs
    for problem in problems

    println("========")
    println("Q = $(Q)")
    println(problem)

    𝒟_train = LearnConvection.Data.data(train, problem; D=32, N=4);
    𝒟_validate = LearnConvection.Data.data(validate, problem; D=32, N=4);
    𝒟_test = LearnConvection.Data.data(test, problem; D=32, N=4);

    problem = Slack("TKE"; parameters=TKEMassFlux.TKEParameters())

    optimized_kernel = LearnConvection.optimize_SMP_kernel(𝒟_train, 𝒟_validate, 𝒟_test; Q=Q)
    𝒢 = LearnConvection.GaussianProcess.model(𝒟_test; kernel=optimized_kernel)
    anim = animate_profile(𝒢, 𝒟_test)

    gif(anim,"SMP_$(typeof(problem))_$(problem.type)_Q_$(Q)_interpolation.gif")
end
end

## Spectral Mixture Product

problem = Slack("TKE"; parameters=TKEMassFlux.TKEParameters())
Q = 1
hyp = [0.18402255361790298, 0.0034913174201650103, 0.001675888689295688]

problem = Slack("KPP"; parameters=KPP.Parameters())
Q = 1
hyp = [0.21342375476405787, 0.03326860797873657, 0.00017668846294285946]

𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel=LearnConvection.GaussianProcess.SpectralMixtureProductI(hyp))
anim = animate_profile_and_model_output(𝒢, 𝒟_test)

𝒟_train = LearnConvection.Data.data(train, problem; D=32, N=4);
𝒟_test = LearnConvection.Data.data(test, problem; D=32, N=4);

gif(anim,"here$(typeof(problem))_$(problem.type)_Q_$(Q)_interpolation.gif")
