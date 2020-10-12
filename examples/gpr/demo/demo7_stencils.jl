using LearnConvection
using Plots

D=32
N=4

train = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2"]
test = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2"]

problem  = Slack("TKE"; parameters=TKEMassFlux.TKEParameters())

train = ["general_strat_4_profiles.jld2", "general_strat_8_profiles.jld2", "general_strat_12_profiles.jld2", "general_strat_16_profiles.jld2"]
test = ["general_strat_28_profiles.jld2", "general_strat_32_profiles.jld2"]

k = 3
logγ = -1.7
distance = euclidean_distance
kernel   = get_kernel(k, logγ, 0.0, distance)

𝒟_train  = LearnConvection.Data.data(train, problem; D=D, N=N);
𝒟_test   = LearnConvection.Data.data(test, problem; D=D, N=N);

all_errors = zeros(D,D)

for stencil_size=1:D
    𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel = kernel, stencil_size = stencil_size)
    all_errors[stencil_size,:] = disparity_vector(𝒢, 𝒟_test)
end

p = plot(all_errors', yscale=:log10, legend=false, xlabel="Stencil size", ylabel="Error", title="Slack(TKE): Error for each gridpoint")

ns = 1:4:32
p = plot(size=(900,600))
for n in ns
    label = 𝒟_train.zavg[n]
    plot!(all_errors[:,n], linewidth=3, yscale=:log10, c=cgrad(:acton)[n*8], xlabel="Stencil size", ylabel="Error", label=label, title="Slack(TKE): Error at different depths")
end
p
png(p,"error_diff_depths2.png")

ns = 1:4:32
p = plot(size=(900,600))
for n in ns
    label = 𝒟_train.zavg[n]
    plot!(collect(17:32), all_errors[17:32,n], linewidth=3, c=cgrad(:acton)[n*8], xlabel="Stencil size", ylabel="Error", label=label, title="Slack(TKE): Error at different depths")
end
p
png(p,"error_diff_depths_linear.png")

mean_errors = [sum(all_errors[stencil_size,:])/D for stencil_size in 1:D]
s = plot(mean_errors, yscale=:log10, legend=false, xlabel="Stencil size", ylabel="Error", title="Slack(TKE): Mean error")
png(s,"mean_error.png")
s2 = plot(collect(16:32), mean_errors[16:32], legend=false, xlabel="Stencil size", ylabel="Error", title="Slack(TKE): Mean error")
png(s2,"mean_error_linear.png")

# png(anim, "error_vs_stencil_size.gif")

# anim = animate_profile(𝒢, 𝒟_test)
anim = animate_profile(𝒢, 𝒟_train)

anim = animate_profile_and_model_output(𝒢, 𝒟_train)
gif(anim, "all.gif")
