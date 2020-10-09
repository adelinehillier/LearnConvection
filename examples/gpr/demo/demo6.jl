
using LearnConvection
using Plots

train = ["general_strat_4_profiles.jld2", "general_strat_32_profiles.jld2"]
validate = ["general_strat_12_profiles.jld2", "general_strat_24_profiles.jld2"]
test = ["general_strat_8_profiles.jld2", "general_strat_16_profiles.jld2", "general_strat_20_profiles.jld2", "general_strat_28_profiles.jld2"]

Qs = [2]
problems = [
    Slack("KPP"; parameters=KPP.Parameters()),
    Slack("TKE"; parameters=TKEMassFlux.TKEParameters())
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

    gif(anim,"$(typeof(problem))_$(problem.type)_Q_$(Q)_interpolation.gif")
end
end


# 𝒟_test = LearnConvection.Data.data(test, problem; D=32, N=4);
#
#
# problem = Slack("TKE"; parameters=TKEMassFlux.TKEParameters())
#
# problem = Slack("KPP"; parameters=KPP.Parameters()),
#
# optimized_kernel = LearnConvection.optimize_SMP_kernel(𝒟_train, 𝒟_validate, 𝒟_test; Q=1)
# 𝒢 = LearnConvection.GaussianProcess.model(𝒟_test; kernel=optimized_kernel)
# anim = animate_profile(𝒢, 𝒟_test)
# gif(anim,"$(typeof(problem))_$(problem.type)_Q_$(Q)_interpolation.gif")
