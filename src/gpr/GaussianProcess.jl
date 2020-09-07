"""
Includes all useful functions for applying GPR to T and wT profiles from Oceananigans.jl simulations.
Uses ProfileData struct to store data and GP struct for performing GPR on the data in ProfileData object.
"""
module GaussianProcess

using LearnConvection.Data

include("kernels.jl")
export  Kernel,
        SquaredExponentialI,
        RationalQuadraticI,
        Matern12I,
        Matern32I,
        Matern52I,
        kernel_function

include("distances.jl")
export  euclidean_distance,
        derivative_distance,
        antiderivative_distance

include("gp.jl")
export  model_output,
        uncertainty,
        compute_kernel_matrix,
        mean_log_marginal_loss
export  model,
        predict

include("errors.jl")
export  get_me_true_check, # evolving forward from an arbitrary initial timestep
        get_me_greedy_check # how well does the mean GP prediction fit the training data?

# plot hyperparameter landscapes for analysis / optimization
include("hyperparameter_landscapes.jl")
export  plot_landscapes_compare_error_metrics,
        plot_landscapes_compare_files_me,
        plot_error_histogram,
        get_min_gamma,
        get_min_gamma_alpha,
        train_validate_test

include("plot_profile.jl")
export  plot_profile,
        plot_model_output,
        animate_profile,
        animate_profile_and_model_output

export get_kernel

"""
# kernel options
#  1   =>   "Squared exponential"         => "Squared exponential kernel:        k(x,x') = σ * exp( - ||x-x'||² / 2γ² )",
#  2   =>   "Matern 1/2"                  => "Matérn with ʋ=1/2:                 k(x,x') = σ * exp( - ||x-x'|| / γ )",
#  3   =>   "Matern 3/2"                  => "Matérn with ʋ=3/2:                 k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)",
#  4   =>   "Matern 5/2"                  => "Matérn with ʋ=5/2:                 k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)",
#  5   =>   "Rational quadratic w/ α=1"   => "Rational quadratic kernel:         k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",
"""
function get_kernel(kernel_id::Int64, logγ, logσ, distance; logα=0.0)
        # convert from log10 scale
        γ = 10^logγ
        σ = 10^logσ
        α = 10^logα
        if kernel_id==1; return SquaredExponentialI(γ, σ, distance) end
        if kernel_id==2; return Matern12I(γ, σ, distance) end
        if kernel_id==3; return Matern32I(γ, σ, distance) end
        if kernel_id==4; return Matern52I(γ, σ, distance) end
        if kernel_id==5; return RationalQuadraticI(γ, σ, α, distance)
        else; throw(error()) end
end


end
