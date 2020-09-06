using JLD2, NetCDF, Statistics, LinearAlgebra, Plots
include("kernels.jl")
include("gaussian_process.jl")
include("../les/get_les_data.jl")
include("../les/custom_avg.jl")

using Plots; pyplot()

filename = "general_strat_16_profiles.jld2"
data = get_les_data(filename);

V = data.T
t = data.t
Nz, Nt = size(V)

D = 16 # gridpoints
zavg = custom_avg(data.z, D) # compress z vector to D values
vavg = [custom_avg(V[:,j], D) for j in 1:Nt] # compress variable array to D values per time
x = vavg[1:(Nt-1)] # (v₀, v₁, ... ,v_(Nt-1)) (Nt-1)-length array of D-length inputs
y = vavg[2:Nt]     # (v₁, v₂, ... ,v_Nt    ) (Nt-1)-length array of D-length targets

# reserve 25% of data for training, but across the entire time interval
total_set = 1:(Nt-1)
training_set = 1:4:(Nt-1)
verification_set = setdiff(total_set, training_set)

n = length(training_set) # num. training points
x_train = x[training_set]
y_train = y[training_set]

# function get_mean_loss(kernel_obj)
#     𝒢 = construct_gpr(x_train, y_train, kernel_obj)
#     return mean_log_marginal_loss(y_train,𝒢)
# end

# get_mean_loss(SquaredExponentialKernelI(1000.0,1.0))
# get_mean_loss(Matern12I(1000.0,1.0))
# get_mean_loss(Matern32I(1000.0,1.0))
# get_mean_loss(Matern52I(1000.0,1.0))

# σs = (1.0:0.1:2)*1.0
# γs = (500:20:1000)*1.0
# Z = [get_mean_loss(SquaredExponentialKernelI(γ,σ)) for γ in γs, σ in σs]
# hplot_exponential = surface(σs,γs,Z,
#                     xlabel="signal variance",
#                     ylabel="squared length scale",
#                     title="log marginal likelihood")

"""
surface plot- hyperparameter landscape
"""
function get_mean_loss(γ,σ)
    kernel = SquaredExponentialKernelI(γ,σ)
    𝒢 = construct_gpr(x_train, y_train, kernel)
    return mean_log_marginal_loss(y_train,𝒢)
end

σs = (0.1:0.1:2)*1.0
γs = (1:5:1000)*1.0
plot(γs,σs,get_mean_loss,st=:surface,c=cgrad([:red,:blue]),camera=(-30,30))

"""
surface plot- hyperparameter landscape
"""
Nt = length(data.t)
set = 1:(Nt-2)
gpr_prediction = similar(y[total_set])
gpr_prediction[1] = x[1] # starting
Nt = length(y[total_set])

# function get_mean_max_error(γ,σ)
#     kernel = SquaredExponentialKernelI(γ,σ)
#     𝒢 = construct_gpr(x_train, y_train, kernel)
#
#
#     for i in 1:(Nt-2)
#         gpr_prediction[i+1] = prediction([gpr_prediction[i]], 𝒢)
#     end
#
#     maxes = zeros(Nt-1)
#     for i in 1:500
#         exact    = y[i+1]
#         predi    = gpr_prediction[i+1]
#         # println(exact)
#         # println(predi)
#         maxes[i] = maximum((exact - predi).^2)
#     end
#
#     return maximum(maxes)
# end
#
# σs = (1.0:0.1:1.5)*1.0
# γs = (500:50:1000)*1.0
# plot(γs,σs,get_mean_max_error,st=:surface,c=cgrad([:red,:blue]),camera=(-30,30))
