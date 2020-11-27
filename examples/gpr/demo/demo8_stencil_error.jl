using LearnConvection
using Plots

D=32
N=4

function animate_error(ℳ, 𝒟)

    xlims = (-0.004, 0.004)
    predi = predict(ℳ, 𝒟; postprocessed=true)

    animation_set = 1:30:(𝒟.Nt-2)
    anim = @animate for i in animation_set
        day_string = string(floor(Int, 𝒟.t[i]/86400))
        plot(𝒟.vavg[i].-predi[i], 𝒟.zavg, legend = :topleft, label = "LES - GP", xlabel = "Temperature [C]", ylabel = "Depth [m]", title = "Error (LES-GP) with stencil size 3, day " * day_string, xlims=xlims)
    end

    return anim
end

train = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2"]
test = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2"]

problem  = Slack("TKE"; parameters=TKEMassFlux.TKEParameters())
# problem  = Sequential("dT")

train = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2", "general_strat_12_profiles.jld2", "general_strat_16_profiles.jld2"]
test = ["general_strat_28_profiles.jld2", "general_strat_32_profiles.jld2"]

k = 3
logγ = -1.7
distance = euclidean_distance
kernel   = get_kernel(k, logγ, 0.0, distance)

𝒟_train  = LearnConvection.Data.data(train, problem; D=D, N=N);
𝒟_test   = LearnConvection.Data.data(test, problem; D=D, N=N);

stencil_size=3
ℳ = LearnConvection.GaussianProcess.model(𝒟_train; kernel = kernel, stencil_size = stencil_size)

anim = animate_error(ℳ, 𝒟_train)

gif(anim, "all.gif")
