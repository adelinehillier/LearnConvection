
using LearnConvection
using Plots

D=16
N=4

## Interpolation

train = ["general_strat_4_profiles.jld2", "general_strat_32_profiles.jld2"]
validate = ["general_strat_12_profiles.jld2", "general_strat_24_profiles.jld2"]
test = ["general_strat_8_profiles.jld2", "general_strat_16_profiles.jld2", "general_strat_20_profiles.jld2", "general_strat_28_profiles.jld2"]


for problem in [Sequential("T"), Sequential("dT"), Residual("KPP", KPP.Parameters()), Residual("TKE", TKEMassFlux.TKEParameters()) ]

    println("$(problem)")

    𝒟_train     = LearnConvection.Data.data(train, problem; D=D, N=N)
    𝒟_validate  = LearnConvection.Data.data(validate, problem; D=D, N=N)
    𝒟_test      = LearnConvection.Data.data(test, problem; D=D, N=N)

    train_validate_test(𝒟_train, 𝒟_validate, 𝒟_test, problem; log_γs=-0.4:0.1:0.4)
end

# 𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel=get_kernel(2,0.0,0.0,euclidean_distance))
# predict(𝒢, 𝒟_train; postprocessed=true)
# LearnConvection.GaussianProcess.get_me_true_check(𝒢, 𝒟_validate)
#
# k=5
# distance=euclidean_distance
# get_min_gamma(k, distance, 𝒟_train, 𝒟_validate, 𝒟_test; log_γs=-0.4:0.1:0.4)
# get_min_gamma_alpha(5, distance, 𝒟_train, 𝒟_validate, 𝒟_test; log_γs=-0.4:0.1:0.4)

## Extrapolation

train = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2", "general_strat_12_profiles.jld2", "general_strat_16_profiles.jld2"]
validate = ["general_strat_20_profiles.jld2", "general_strat_24_profiles.jld2"]
test = ["general_strat_28_profiles.jld2", "general_strat_32_profiles.jld2"]

for problem in [Sequential("T"), Sequential("dT"), Residual("KPP", KPP.Parameters()), Residual("TKE", TKEMassFlux.TKEParameters()) ]

    println("$(problem)")

    𝒟_train     = LearnConvection.Data.data(train, problem; D=D, N=N)
    𝒟_validate  = LearnConvection.Data.data(validate, problem; D=D, N=N)
    𝒟_test      = LearnConvection.Data.data(test, problem; D=D, N=N)

    train_validate_test(𝒟_train, 𝒟_validate, 𝒟_test, problem; log_γs=-0.4:0.1:0.4)
end
