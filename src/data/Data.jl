"""
Data module for preparing data for analysis with
    - GaussianProcess (src/gpr/GaussianProcess.jl)
"""

module Data

using OceanTurb
using OrderedCollections

include("../les/custom_avg.jl")

# harvesting Oceananigans data
include("../les/get_les_data.jl")
export get_les_data

# normalization
include("scalings.jl")
# export  Tscaling,
#         wTscaling
export  min_max_scaling,
        scale, # normalize
        unscale # un-normalize

# pre- and post-processing on the normalized data
include("problems.jl")
export  Problem,
        Sequential,
        Residual,
        Slack,
        SequentialProblem,
        ResidualProblem,
        SlackProblem,
        get_problem_v,
        predict # important!

export Sequential_KPP,
       Sequential_TKE

include("sequential.jl")
include("residual.jl")
include("slack.jl")
export  get_predictors_targets,
        postprocess_prediction

# running OceanTurb KPP simulations based on OceananigansData conditions
include("../kpp/run.jl")
export closure_free_convection_kpp_full_evolution,
       closure_free_convection_kpp
# running OceanTurb TKE simulations based on OceananigansData conditions
include("../tke/run.jl")
export closure_free_convection_tke_full_evolution,
       closure_free_convection_tke

# ProfileData struct
export  ProfileData,
        data

function approx_initial_buoyancy_stratification(T_initial,z)
    # b_initial = 𝒟.T[:,1] .* α*g
    α = 2e-4
    g = 9.80665
    N² = (T_initial[1] - T_initial[end])*α*g / z[1] # approximate initial buoyancy gradient N², where b = N²z + 20*α*g and T = N²z/(αg) + 20
    return N²
end

"""
ProfileData
# Description
- data structure for preparing profile data for analysis with gpr or nn.
# Data Structure and Description
    v::Array,           Nz x Nt array of T or wT values directly from the LES simulation, not preprocessed.
    vavg::Array,        Nt-length array of Nz-length vectors from LES simulation, scaled and pre-processed
    x::Array,           all simulation inputs, scaled and pre-processed
    y::Array,           all simulation inputs, scaled and pre-processed
    x_train::Array,     training inputs (predictors; array of states). (length-n array of D-length vectors, where D is the length of each input n is the number of training points)
    y_train::Array,     training outputs (predictions) (length-n array of D-length vectors).
    validation_set::Array, vector of indices corresponding to validation data
    z::Array,           Nz-length vector of depth values
    zavg::Array,        length-D vector; depth values averaged to D gridpoints
    t::Array,           timeseries [seconds]
    Nt::Int64,          length(timeseries)
    n_train::Int64,     number of training pairs
    κₑ::Float,          eddy diffusivity
    processor::DataProcessor, struct for preparing the data for GP regression
    problem::Problem,   what mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"), Residual("T"), Residual("KPP"), or Residual("TKE"))

"""
struct ProfileData
    v       ::Array{Float64,2}
    vavg    ::Array{Array{Float64,1},1}
    x       ::Array{Array{Float64,1},1}
    y       ::Array{Array{Float64,1},1}
    x_train ::Array{Array{Float64,1},1}
    y_train ::Array{Array{Float64,1},1}
    validation_set::Vector{Int64}
    z       ::Vector{Float64}
    zavg    ::Vector{Float64}
    t       ::Vector{Float64}
    Nt      ::Int64
    # n_train ::Int64
    # κₑ      ::Float64
    problem ::Problem
    all_problems::Array{Array{Any,1},1}
end

"""
data(filename, problem; D=16, N=4)

# Description
Returns a ProfileData object based on data from `filename`

# Arguments
- `filename`: (string)  Name of the NetCDF or JLD2 file containing the data from the Oceananigans simulation.
- `problem`: (Problem). What mapping you wish to evaluate with the model.

# Keyword Arguments
- `D`: (integer)        Number of gridpoints in the z direction to average the data to for training and prediction.
- `N`: (integer)        Interval between the timesteps to be reserved for training data (default 4).
                        If N=4, the profile data for every 4 timesteps will be reserved for training (~25% training data);
                        the rest will be used in the validation set.
"""
function data(filename::String, problem::Problem; D=16, N=4)

    # collect data from Oceananigans simulation output file
    les = get_les_data(filename) # <: OceananigansData

    # eddy diffusivity
    κₑ = les.κₑ

    # depth values
    z = les.z
    zavg = custom_avg(z, D)

    # approximate buoyancy stratification at the initial timestep
    N² = approx_initial_buoyancy_stratification(les.T[:,1],z)

    # get v (variable array, Nz x Nt) and cut out the first 2 hours
    s = Int(120 / (les.t[2] - les.t[1])):length(les.t)
    v = get_v(problem)[s]

    # timeseries [s]
    t = les.t[s]
    Nt = length(t)

    # get problem (sets how the data will be pre- and post-processed)
    problem = get_problem(problem, les, v, N², D)

    # compress variable array to D gridpoints to get an Nt-length array of D-length vectors
    vavg = [custom_avg(v[:,j], D) for j in 1:Nt]

    # divide up the data
    total_set = 1:(Nt-1)
    training_set = 1:N:(Nt-1)
    validation_set = setdiff(total_set, training_set)
    n_train = length(training_set)

    # get the predictors and target predictions to train the model on
    x, y = get_predictors_targets(vavg, problem)

    x_train = x[training_set]
    y_train = y[training_set]

    # if the data contains multiple files, we'll need a way to postprocess the data separately for each one
    # (preprocessing is already taken care of before the data is merged)
    all_problems = [[problem, length(x)]]

    return ProfileData(v, vavg, x, y, x_train, y_train, validation_set, z, zavg, t, Nt, problem, all_problems)
end

"""
data(filename, D; N=4)

# Description
Returns an instance of ProfileData containing training data from multiple simulations.
*** Important:
    ONLY v, x_train and y_train contain data from all filenames;
    the remaining attributes are taken from the first filename in filenames

# Arguments
- `filenames`: (string)  Vector of filenames (.nc or .jld2) to collect data from.
- `problem`: (Problem). What mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"), Residual("T"), Residual("KPP"), or Residual("TKE"))

# Keyword Arguments
- `D`: (integer)        Number of gridpoints in the z direction to average the data to for training and prediction.
- `N`: (integer)        Interval between the timesteps to be reserved for training data (default 4).
                        If N=4, the profile data for every 4 timesteps will be reserved for training (~25% training data);
                        the rest will be used in the validation set.
"""
function data(filenames::Vector{String}, problem::Problem; D=16, N=4)

    # combines data from multiple files
    𝒟 = data(filenames[1], problem; D=D, N=N)

    v = 𝒟.v
    vavg = 𝒟.vavg
    x = 𝒟.x
    y = 𝒟.y
    x_train = 𝒟.x_train
    y_train = 𝒟.y_train
    validation_set = 𝒟.validation_set
    all_problems = 𝒟.all_problems

    t = 𝒟.t
    Nt = 𝒟.Nt

    for filename in filenames[2:end]
        𝒟2 = data(filename, problem; D=D, N=N)

        v = hcat(v, 𝒟2.v) # unscaled
        vavg = vcat(vavg, 𝒟2.vavg)
        x = vcat(x, 𝒟2.x)
        y = vcat(y, 𝒟2.y)
        validation_set = vcat(validation_set, 𝒟2.validation_set)
        x_train = vcat(x_train, 𝒟2.x_train)
        y_train = vcat(y_train, 𝒟2.y_train)
        append!(all_problems, 𝒟2.all_problems)
        t = vcat(t, 𝒟2.t)
        Nt += 𝒟2.Nt
    end

    # Note the problem is that from the first file in filenames. This is only included so that the problem type can be determined easily.
    return ProfileData(v, vavg, x, y, x_train, y_train, validation_set, 𝒟.z, 𝒟.zavg, t, Nt, 𝒟.problem, all_problems)
end

end # module
