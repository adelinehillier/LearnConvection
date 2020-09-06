
# """
# OLD version: uses gaussian_process.jl directly. New version uses data.jl to prepare the data.
# Interactive exploration of the hyperparameter space using Interact.jl.
# """

using JLD2, Statistics, LinearAlgebra, Plots
using Interact, Blink

include("kernels.jl")
include("gaussian_process.jl")
include("../les/get_les_data.jl")

# file to gather data from
filename = togglebuttons(OrderedDict("general_strat_16_profiles" =>"general_strat_16_profiles.jld2",
                                     "general_strat_32_profiles" =>"general_strat_32_profiles.jld2"),
                                     label="LES")

D = 16 # gridpoints

# these variables are common to the files in this example
# -- CHANGE if using different files!! --
data = get_les_data(filename[])
t = data.t
Nt = length(t)

total_set = 1:(Nt-1)
training_set = 1:4:(Nt-1)
verification_set = setdiff(total_set, training_set)

# which variable to explore
V_name = togglebuttons(Dict("T" =>"Temperature [°C]", "wT"=>"Temperature flux [°C⋅m/s]"), label="profile")
# smooth the profile
smooth = toggle("smooth profile?")
# normalize the data (pre- / postprocessing)
normalize = toggle("normalize?")

γs = -3.0:0.1:3.0
σs = 0.0:0.1:2.0

# hyperparameter knobs
γ1 = slider(γs, label="log length scale, log₁₀(γ)")
σ1 = slider(σs, label="log signal variance, log₁₀(σ²)")
time_index = slider(1:40:(Nt-1), label="time [s]")

dist_metric = tabulator(OrderedDict("l²-norm"  =>  "l²-norm:  d(x,x') = || x - x' ||",
                                    "H¹-norm"  =>  "H¹-norm:  d(x,x') = || diff(x)./diff(z) - diff(x')./diff(z) ||",
                                    "H⁻¹-norm" =>  "H⁻¹-norm: d(x,x') = || diff(x).*diff(z) - diff(x').*diff(z) ||"
                                    ))

kern = tabulator(OrderedDict("Squared exponential"          => "Squared exponential kernel:           k(x,x') = σ * exp( - ||x-x'||² / 2γ² )",
                             "Matern 1/2"                    => "Matérn with ʋ=1/2:                    k(x,x') = σ * exp( - ||x-x'|| / γ )",
                             "Matern 3/2"                    => "Matérn with ʋ=3/2:                    k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)",
                             "Matern 5/2"                    => "Matérn with ʋ=5/2:                    k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)",
                             "Rational quadratic w/ α=1"     => "Rational quadratic kernel:            k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",
                            ))

get_data(filename::String) = get_les_data(filename)

function get_V(V_name, data)
    if V_name=="Temperature [°C]"
        return data.T
    end
    if V_name=="Temperature flux [°C⋅m/s]"
        return data.wT
    else
        throw(error())
    end
end

function get_d(dist_metric)
    if dist_metric==1; return l2_norm end
    if dist_metric==2; return h1_norm end
    if dist_metric==3; return hm1_norm
    else
        throw(error())
    end
end

get_vavg(V)       = [custom_avg(V[:,j], D) for j in 1:Nt] # compress variable array to D values per time
get_x(vavg)       = vavg[1:(Nt-1)] # (v₀, v₁, ... ,v_(Nt-1)) (Nt-1)-length array of D-length inputs
get_y(vavg)       = vavg[2:(Nt)]   # (v₁, v₂, ... ,v_Nt    ) (Nt-1)-length array of D-length targets
get_x_train(x)    = x[training_set]
get_y_train(y)    = y[training_set]
get_z(data)       = custom_avg(data.z, D)

function get_kernel(k::Int64, γ, σ)
    # convert from log10 scale
    γ = 10^γ
    σ = 10^σ
  if k==1; return SquaredExponentialKernelI(γ, σ) end
  if k==2; return Matern12I(γ, σ) end
  if k==3; return Matern32I(γ, σ) end
  if k==4; return Matern52I(γ, σ) end
  if k==5; return RationalQuadraticI(γ, σ, 1.0)
  else; throw(error()) end
end

function plot_kernel(kernel::Kernel, d, z)
    kmat = [kernel_function(kernel; d=d, z=z)(i,j) for i in 1:10:Nt, j in 1:10:Nt]# fill kernel mx with values
    return heatmap(kmat, title = "Covariance Matrix", xaxis=(:false), yaxis=(:flip, :false), clims=(0.0,100), legend=true)
end

function get_gp(x_train, y_train, kernel::Kernel, normalize, d, z)
    𝒢 = construct_gpr(x_train, y_train, kernel; distance_fn=d, z=z, normalize=normalize);
    return 𝒢
end

function get_gpr_pred(gp::GP, x, vavg)
    # predict temperature profile from start to finish without the training data
    gpr_prediction = similar(vavg[1:Nt-1])
    starting = x[1]
    gpr_prediction[1] = starting
    for i in 1:(Nt-2)
        gpr_prediction[i+1] = prediction([gpr_prediction[i]], gp)
    end
    return gpr_prediction
end

function plot_profile(gp::GP,V,V_name,time_index,gpr_prediction,z)
    exact = V[:,time_index+1]
    day_string = string(floor(Int, t[time_index]/86400))
    p = scatter(gpr_prediction[time_index+1], z, label = "GP")
    plot!(exact, z, legend = :topleft, label = "LES", xlabel = "$(V_name)", ylabel = "depth", title = "day " * day_string)
    return p
end

function plot_log_error(gp::GP, time_index, x, y, vavg)
    # gpr_error = collect(verification_set)*1.0
    # # greedy check
    # for j in eachindex(verification_set)
    #     test_index = verification_set[j]
    #     y_prediction = prediction([x[test_index]], gp)
    #     error = norm(y_prediction - y[test_index])
    #     gpr_error[j] = error
    # end
    # mean_error = sum(gpr_error)/length(gpr_error)
    # error_plot_log = histogram(log.(gpr_error), title = "log(Error)", xlabel="log(Error)", ylabel="Frequency",xlims=(-20,0),ylims=(0,250), label="frequency")
    # vline!([log(mean_error)], line = (4, :dash, 0.8), label="mean error")
    # vline!([log(gpr_error[time_index])], line = (1, :solid, 0.6), label="error at t=$(time_index)")

    #compute error for true check
    gpr_error = zeros(Nt-2)
    gpr_prediction = get_gpr_pred(gp, x, vavg)
    for i in 1:Nt-2
        exact    = y[i+1]
        predi    = gpr_prediction[i+1]
        gpr_error[i] = l2_norm(exact, predi) # euclidean distance
    end
    mean_error = sum(gpr_error)/(Nt-2)

    error_plot_log = histogram(log.(gpr_error), title = "log(error) at each timestep of the full evolution", xlabel="log(Error)", ylabel="Frequency",xlims=(-20,0),ylims=(0,250), label="frequency")
    vline!([log(mean_error)], line = (4, :dash, 0.8), label="mean error")
    vline!([log(gpr_error[time_index])], line = (1, :solid, 0.6), label="error at t=$(time_index)")

end

function plot_hyp_landscp(k::Int64, x_train, y_train, x, y, vavg, normalize, d, z)

    mlls = zeros(length(γs)) # mean log marginal likelihood
    mes  = zeros(length(γs)) # mean error (greedy check)
    mets  = zeros(length(γs)) # mean error (true check)

    for i in 1:length(γs)
        kernel = get_kernel(k, γs[i], 0.0)
        𝒢 = construct_gpr(x_train, y_train, kernel; distance_fn=d, z=z, normalize=normalize);
        mlls[i] = -1*mean_log_marginal_loss(y_train, 𝒢::GP, add_constant=false)

        #compute mean error for greedy check (same as in plot log error)
        total_error = 0.0
        # greedy check
        for j in eachindex(verification_set)
            test_index = verification_set[j]
            y_prediction = prediction([x[test_index]], 𝒢)
            error = l2_norm(y_prediction, y[test_index])
            total_error += error
        end
        mes[i] = total_error/length(verification_set)

        total_error = 0.0
        #compute mean error for true check
        gpr_prediction = get_gpr_pred(𝒢, x, vavg)
        for i in 1:Nt-2
            exact    = y[i+1]
            predi    = gpr_prediction[i+1]
            total_error += l2_norm(exact, predi) # euclidean distance
        end
        mets[i] = total_error/(Nt-2)

    end

    mll_plot = plot(γs, mlls, xlabel="log(γ)", title="negative mean log marginal likelihood, P(y|X)", legend=false, yscale=:log10) # 1D plot: mean log marginal loss vs. γ
    # me_plot  = plot(γs, mets,  xlabel="log(γ)", title="mean error on full evolution ('true check'), min = $(minimum(mets))", legend=false, yscale=:log10)  # 1D plot: mean error vs. γ
    me_plot  = plot(γs, mets,  xlabel="log(γ)", title="min = $(minimum(mets))", legend=false, yscale=:log10)  # 1D plot: mean error vs. γ

    return plot(mll_plot, me_plot, layout = @layout [a ; b])
end


#updating variables
#output               function          args
data            = map(get_data,         filename)
d               = map(get_d,            dist_metric)
V               = map(get_V,            V_name, data)
vavg            = map(get_vavg,         V)
x               = map(get_x,            vavg)
y               = map(get_y,            vavg)
z               = map(get_z,            data)
x_train         = map(get_x_train,      x)
y_train         = map(get_y_train,      y)
k               = map(get_kernel,       kern, γ1, σ1)
k_plot          = map(plot_kernel,      k, d, z)
gp              = map(get_gp,           x_train, y_train, k, normalize, d, z)
gpr_prediction  = map(get_gpr_pred,     gp, x_train, vavg)
profile_plot    = map(plot_profile,     gp, V, V_name, time_index, gpr_prediction, z)
log_error_plot  = map(plot_log_error,   gp, time_index, x, y, vavg)
hyp_landscape   = map(plot_hyp_landscp, kern, x_train, y_train, x, y, vavg, normalize, d, z)

println(normalize[])

# layout
top    = vbox(hbox(filename, V_name, vbox(smooth, normalize)), hbox(kern, dist_metric), hbox(k_plot, hyp_landscape))
middle = vbox(γ1, σ1, time_index)
bottom = hbox(profile_plot, log_error_plot)
ui = vbox(top, middle, bottom) # aligns vertically

# Blink GUI
window = Window()
body!(window, ui)
