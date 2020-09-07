
using LearnConvection
using Plots

D=16
N=8

problem  = Sequential("dT")

train = ["general_strat_4_profiles.jld2", "general_strat_32_profiles.jld2"]
𝒟_train = LearnConvection.Data.data(train, problem; D=D, N=N)
𝒟_validate  = LearnConvection.Data.data("general_strat_16_profiles.jld2", problem; D=D, N=N)
𝒟_test = LearnConvection.Data.data("general_strat_32_profiles.jld2", problem; D=D, N=N)

𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel=get_kernel(4,0.0,0.0,euclidean_distance))
predict(𝒢, 𝒟_train; postprocessed=true)
LearnConvection.GaussianProcess.get_me_true_check(𝒢, 𝒟_validate)

k=3
distance=euclidean_distance
get_min_gamma(k, distance, 𝒟_train, 𝒟_validate, 𝒟_test; log_γs=-0.4:0.1:0.4)


##






using LearnConvection
using Plots

# Construct a ProfileData object, 𝒟, consisting of the data to train on,
# and a GP object, 𝒢, with which to perform the regression.

# data
filename = "general_strat_32_profiles.jld2"
problem  = Residual("KPP", KPP.Parameters()) # v: "T" for temperature profile; "wT" for temperature flux profile
problem  = Residual("TKE", TKEMassFlux.TKEParameters())
D        = 32       # collapse profile data down to 16 gridpoints
N        = 4        # collect every 4 timesteps' data for training

# Now let's define our kernel. We don't know what the best long(length-scale) parameter value is yet so let's guess 0.0 (γ=1) and see what happens.
# Let's use an exponential kernel.

# kernel
k        = 3        # kernel function ID
logγ     = -0.6      # log(length-scale parameter)
logσ     = 0.0      # log(signal variance parameter)
distance = derivative_distance  # distance metric to use in the kernel
kernel   = get_kernel(k, logγ, logσ, distance)

# data
𝒟_train = LearnConvection.Data.data(["general_strat_16_profiles.jld2","general_strat_32_profiles.jld2"], problem; D=D, N=N)
𝒟_test = LearnConvection.Data.data("general_strat_32_profiles.jld2", problem; D=D, N=N)

# model
𝒢 = LearnConvection.GaussianProcess.model(𝒟_train; kernel = kernel)

# Animate the mean GP prediction.
# anim = animate_profile(𝒢, 𝒟)
# gif(anim, "animated_profile.gif")

anim = animate_profile_and_model_output(𝒢, 𝒟_test)
gif(anim, "animated_profile_and_model_output_KPP32_train_gs16_gs32_test_gs32.gif")

##

# Not great. Let's try to optimize our γ value.
# We can use get_min_gamma function to search the range -2:0.1:2 for the log(γ) value that minimizes the mean error on the true check
min_log_param, min_error = get_min_gamma(k, 𝒟, distance, -2:0.01:2)

# Let's create a kernel with the better log(γ) value, put the new kernel into a new model, and see how it does.
new_kernel = get_kernel(k, min_log_param, logσ, distance)
𝒢 = LearnConvection.GaussianProcess.model(𝒟; kernel = new_kernel)
anim = animate_profile(𝒢, 𝒟)
gif(anim, "animated_profile_optimized.gif")

# There's a lot of flexibility in how we design our kernel function.
# We can modify the hyperparameter values, the distance metric, and the type of kernel function.

p1 = plot([1 2 3], [4 5 6], title="hello", ylabel="yes")
p2 = plot([1 2 3], [4 5 6])
p3 = plot([1 2 3], [4 5 6])
p4 = plot([1 2 3], [4 5 6])
p5 = plot([1 2 3], [4 5 6])
p6 = plot([1 2 3], [4 5 6])

layout = @layout [a b c; d e f]
titles = ["Sq Exp / l²-norm" "Matérn 1/2 / l²-norm" "Matérn 3/2 / l²-norm" "Sq Exp / H¹-norm" "Sq Exp / H¹-norm" "Sq Exp / H¹-norm"]

plot(p1, p2, p3, p4, p5, p6; layout = layout, title=titles)






# data
𝒟 = OceanConvect.ModelData.data("general_strat_16_profiles.jld2", Sequential("dT"); D=16, N=4)
logγs = -3.0:0.1:3.0
p1  = plot_landscapes_compare_error_metrics(1, 𝒟, euclidean_distance, log_γs)
p2  = plot_landscapes_compare_error_metrics(2, 𝒟, euclidean_distance, log_γs)
p3  = plot_landscapes_compare_error_metrics(3, 𝒟, euclidean_distance, log_γs)
p4  = plot_landscapes_compare_error_metrics(4, 𝒟, euclidean_distance, log_γs)
p5  = plot_landscapes_compare_error_metrics(5, 𝒟, euclidean_distance, log_γs)
p6  = plot_landscapes_compare_error_metrics(1, 𝒟, derivative_distance, log_γs)
p7  = plot_landscapes_compare_error_metrics(2, 𝒟, derivative_distance, log_γs)
p8  = plot_landscapes_compare_error_metrics(3, 𝒟, derivative_distance, log_γs)
p9  = plot_landscapes_compare_error_metrics(4, 𝒟, derivative_distance, log_γs)
p10 = plot_landscapes_compare_error_metrics(5, 𝒟, derivative_distance, log_γs)
p11 = plot_landscapes_compare_error_metrics(1, 𝒟, antiderivative_distance, log_γs)
p12 = plot_landscapes_compare_error_metrics(2, 𝒟, antiderivative_distance, log_γs)
p13 = plot_landscapes_compare_error_metrics(3, 𝒟, antiderivative_distance, log_γs)
p14 = plot_landscapes_compare_error_metrics(4, 𝒟, antiderivative_distance, log_γs)
p15 = plot_landscapes_compare_error_metrics(5, 𝒟, antiderivative_distance, log_γs)








##

# problems
params   = KPP.Parameters( CSL = 1.0, CNL = 1.0, Cb_T = 1.0, CKE = 1.0)
params   = KPP.Parameters( )

problem  = Residual("KPP", params) # v: "T" for temperature profile; "wT" for temperature flux profile

problem  = Sequential("dT") # v: "T" for temperature profile; "wT" for temperature flux profile

# data
filename = "general_strat_32_profiles.jld2"
D        = 16
N        = 4

# kernel
k        = 1
logγ     = -3.0
logσ     = 0.0
distance = euclidean_distance
kernel   = get_kernel(k, logγ, logσ, distance)

# data
𝒟 = LearnConvection.Data.data(filename, problem; D=D, N=N)

# model
𝒢 = LearnConvection.GaussianProcess.model(𝒟; kernel = kernel)

# Animate the mean GP prediction.
anim = animate_profile(𝒢, 𝒟)
gif(anim, "animated_profile.gif")
