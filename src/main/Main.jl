module Main

using LearnConvection.Data
using LearnConvection.GaussianProcess

include("errors.jl")
export  get_me_true_check, # evolving forward from an arbitrary initial timestep
        get_me_greedy_check # how well does the mean GP prediction fit the training data?

include("plot_profile.jl")
export  plot_profile,
        plot_model_output,
        animate_profile,
        animate_profile_and_model_output

include("predict.jl")
export predict

end
