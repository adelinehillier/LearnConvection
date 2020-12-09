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
        uncertainty,

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

# modules
using Plots,
      JLD2,
      NetCDF,
      Statistics,
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
