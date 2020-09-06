"""
Data module for preparing data for analysis with

    - GaussianProcess (src/gpr/GaussianProcess.jl)
        or
    - NeuralNetwork (src/gpr/NeuralNetwork.jl)

"""

module Data

using OceanTurb

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
        SequentialProblem,
        ResidualProblem,
        get_problem_v

include("residual.jl")
include("sequential.jl")
export  get_predictors_targets,
        postprocess_prediction

# running OceanTurb KPP simulations based on OceananigansData conditions
include("../kpp/run.jl")
export closure_free_convection_kpp
# running OceanTurb TKE simulations based on OceananigansData conditions
include("../tke/run.jl")
export closure_free_convection_tke

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
    v       ::Array
    vavg    ::Array
    x       ::Array
    y       ::Array
    x_train ::Array
    y_train ::Array
    validation_set::Array
    z       ::Array
    zavg    ::Array
    t       ::Array
    Nt      ::Int64
    n_train ::Int64
    κₑ      ::Float64
    problem ::Problem
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

    # timeseries [s]
    t = les.t
    Nt = length(t)

    # eddy diffusivity
    κₑ = les.κₑ

    # depth values
    z = les.z
    zavg = custom_avg(z, D)

    # approximate buoyancy stratification at the initial timestep
    N² = approx_initial_buoyancy_stratification(les.T[:,1],z)

    # get problem (sets how the data will be pre- and post-processed) and v (variable array, Nz x Nt)
    v, problem = get_problem_v(problem, les, N², D, t)

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

    return ProfileData(v, vavg, x, y, x_train, y_train, validation_set, z, zavg, t, Nt, n_train, κₑ, problem)
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
function data(filenames, problem::Problem; D=16, N=4)

Array{Array{Int64,1},1}
    # combines data from multiple files
    𝒟 = data(filenames[1], problem; D=D, N=N)

    v = 𝒟.v
    x_train = 𝒟.x_train
    y_train = 𝒟.y_train

    for filename in filenames[2:end]
        data_b = data(filename, problem; D=D, N=N)

        training_set = 1:N:(data_b.Nt-1)

        v = hcat(v, data_b.v) # unscaled
        x_train = vcat(x_train, data_b.x[training_set])
        y_train = vcat(y_train, data_b.y[training_set])
    end

    # ONLY v, x_train and y_train contain data from all filenames, the rest of the attributes are from the first filename in filenames
    return ProfileData(v, 𝒟.vavg, 𝒟.x, 𝒟.y, x_train, y_train, 𝒟.validation_set, 𝒟.z, 𝒟.zavg, 𝒟.t, 𝒟.Nt, 𝒟.n_train, 𝒟.κₑ, 𝒟.problem)
end

end # module
