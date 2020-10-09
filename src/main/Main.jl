module Main

using LearnConvection.Data
using LearnConvection.GaussianProcess

model_output(x, time_index, ℳ, 𝒟) = GaussianProcess.model_output(𝒟.modify_predictor_fn(x, time_index), ℳ)

include("errors.jl")
export  get_me_true_check, # evolving forward from an arbitrary initial timestep
        get_me_greedy_check, # how well does the mean GP prediction fit the training data?
        disparity_vector

include("plot_profile.jl")
export  plot_profile,
        plot_model_output,
        animate_profile,
        animate_profile_and_model_output

include("predict.jl")
export predict

# plot hyperparameter landscapes for analysis / optimization
include("hyperparameter_landscapes.jl")
export  plot_landscapes_compare_error_metrics,
        plot_landscapes_compare_files_me,
        plot_error_histogram,
        get_min_gamma,
        get_min_gamma_alpha,
        train_validate_test

include("optimize_smp.jl")
export  optimize_SMP_kernel

end #module
