"""
Includes all useful functions for applying GPR to T and wT profiles from Oceananigans.jl simulations.
Uses ProfileData struct to store data and GP struct for performing GPR on the data in ProfileData object.
"""

module LearnConvection

export
        # Data / profile_data.jl
        data,
        ProfileData,
        Sequential,
        Residual,
        Slack,

        # Data / modify_predictor_fns.jl,
        append_tke,
        partial_temp_profile,

        # GaussianProcess / gp.jl
        # construct_gpr,
        uncertainty,
        # model,

        # GaussianProcess / kernels.jl
        Kernel,
        get_kernel,
        kernel_function,

        # GaussianProcess / distances.jl
        euclidean_distance,
        derivative_distance,
        antiderivative_distance,

        # GaussianProcess / hyperparameter_landscapes.jl
        plot_landscapes_compare_error_metrics,
        plot_landscapes_compare_files_me,
        plot_error_histogram,
        get_min_gamma,
        get_min_gamma_alpha,
        train_validate_test,

        # Main / errors.jl
        get_me_true_check,
        get_me_greedy_check,
        disparity_vector,

        # Main / plot_profile.jl
        plot_profile,
        plot_model_output,
        animate_profile,
        animate_profile_and_model_output,

        # Main / predict.jl
        predict,

        # Main / optimize_smp.jl
        optimize_SMP_kernel

        # kernel options
        #  1   =>   "Squared exponential"         => "Squared exponential kernel:        k(x,x') = σ * exp( - ||x-x'||² / 2γ² )",
        #  2   =>   "Matern 1/2"                  => "Matérn with ʋ=1/2:                 k(x,x') = σ * exp( - ||x-x'|| / γ )",
        #  3   =>   "Matern 3/2"                  => "Matérn with ʋ=3/2:                 k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)",
        #  4   =>   "Matern 5/2"                  => "Matérn with ʋ=5/2:                 k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)",
        #  5   =>   "Rational quadratic w/ α=1"   => "Rational quadratic kernel:         k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",

# modules
using Plots,
      JLD2,
      NetCDF,
      Statistics,
      # LinearAlgebra,
      BenchmarkTools,
      Optim

# OceanTurb for KPP
using OceanTurb
export KPP, TKEMassFlux

# submodules
include("data/Data.jl")
include("gpr/GaussianProcess.jl")
include("main/Main.jl")

using .Data
using .GaussianProcess
using .Main

end
