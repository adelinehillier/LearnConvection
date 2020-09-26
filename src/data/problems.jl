"""
Used in gp.jl. Data pre- / post-processing. Takes a ProfileData object and prepares it for use in GP.
"""

abstract type Problem end

abstract type SequentialProblem <: Problem end
abstract type ResidualProblem <: Problem end
abstract type SlackProblem <: Problem end

mutable struct Sequential <: SequentialProblem
    type::String # "T" or "dT" or "wT"
    parameters # Parameters(...) (see OceanTurb documentation)
    Sequential(type) = new(type) # incomplete initialization
    Sequential(type, parameters) = new(type, parameters)
end

mutable struct Residual <: ResidualProblem
    type::String # "KPP" or "TKE"
    parameters # Parameters(...) (see OceanTurb documentation)
end

struct Slack <: SlackProblem
    type::String # "KPP" or "TKE"
    parameters # Parameters(...) (see OceanTurb documentation)
end

mutable struct Sequential_dT <: SequentialProblem
    variable::String #"T" or "wT"
    Δt::Number # assumes constant time interval between all timesteps
    scaling::Scaling
    Sequential_dT(variable, Δt) = new(variable, Δt) # incomplete initialization
end

mutable struct Sequential_T <: SequentialProblem
    variable::String # "T" or "wT"
    scaling::Scaling
    Sequential_T(variable) = new(variable) # incomplete initialization
end

mutable struct Sequential_wT <: SequentialProblem
    variable::String # "T" or "wT"
    scaling::Scaling
    Sequential_wT(variable) = new(variable) # incomplete initialization
end

mutable struct Sequential_KPP <: SequentialProblem
    variable::String
    physics_data
    evolve_physics_model_fn # Function
    Δt # assumes constant time interval between all timesteps
    scaling::Scaling
    Sequential_KPP(variable, physics_data, evolve_physics_model_fn, Δt) = new(variable, physics_data, evolve_physics_model_fn, Δt) # incomplete initialization
end

mutable struct Sequential_TKE <: SequentialProblem
    variable::String
    physics_data
    evolve_physics_model_fn # Function
    Δt # assumes constant time interval between all timesteps
    scaling::Scaling
    Sequential_TKE(variable, physics_data, evolve_physics_model_fn, Δt) = new(variable, physics_data, evolve_physics_model_fn, Δt) # incomplete initialization
end

mutable struct Residual_KPP <: ResidualProblem
    variable::String
    physics_data
    evolve_physics_model_fn # Function
    Δt # assumes constant time interval between all timesteps
    scaling::Scaling
    Residual_KPP(variable, physics_data, evolve_physics_model_fn, Δt) = new(variable, physics_data, evolve_physics_model_fn, Δt) # incomplete initialization
end

mutable struct Residual_TKE <: ResidualProblem
    variable::String
    physics_data
    evolve_physics_model_fn # Function
    Δt # assumes constant time interval between all timesteps
    scaling::Scaling
    Residual_TKE(variable, physics_data, evolve_physics_model_fn, Δt) = new(variable, physics_data, evolve_physics_model_fn, Δt) # incomplete initialization
end

mutable struct Slack_KPP <: SlackProblem
    variable::String
    physics_data::Array
    scaling::Scaling
    Slack_KPP(variable, physics_data) = new(variable, physics_data) # incomplete initialization
end

mutable struct Slack_TKE <: SlackProblem
    variable::String
    physics_data::Array
    scaling::Scaling
    Slack_TKE(variable, physics_data) = new(variable, physics_data) # incomplete initialization
end

"""
get_v(problem)
# Description
    Return data array from the OceananigansData object associated with the variable of interest.
# Arguments
- 'problem': (Problem).     what mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"))
"""
function get_v(problem)
    if problem.type == "wT"
        return les.wT
    elseif problem.type in ["T", "dT", "KPP", "TKE"]
        return les.T
    else
        throw(error())
    end
end

"""
get_problem(problem::Problem, data::OceananigansData, timeseries)
# Description
    Creates an instance of a Problem struct depending on the type of mapping.
    Returns the problem and the correct data array from the OceananigansData objectssociated with the variable of interest.
# Arguments
- 'problem': (Problem).      what mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"))
- 'v': (Array).              Array of values
- 'les': (OceananigansData). LES simulation output
- 'N²': (Number)             initial buoyancy stratification
- 'D': (Number)              number of grid points in the vertical
"""
function get_problem(problem, les, v, N², D)

    Δt = les.t[2]-les.t[1]

    if typeof(problem) <: SequentialProblem

        if problem.type == "dT"
            Δt /= N² # Δt' = Δt / N²
            return Sequential_dT("T", Δt) # Sequential_dT(variable, Δt)

        elseif problem.type == "T"
            return Sequential_T("T")

        elseif problem.type == "wT"
            return Sequential_wT("wT")

        elseif problem.type == "KPP"
            f = closure_free_convection_kpp(D, Δt, les)
            evolve_physics_model_fn(T⁰) = f(problem.parameters; T⁰=T⁰, n_steps=1)[:,2]
            kpp_data = Array{Array{Float64,1},1}(UndefInitializer(), length(les.t))
            kpp_data[1] = custom_avg(v[:,1], D)
            for i in 1:length(les.t)-1
                kpp_data[i+1] = evolve_physics_model_fn(v[:,i])
            end
            return Sequential_KPP("T", kpp_data, evolve_physics_model_fn, Δt)

        elseif problem.type == "TKE"
            # Use the LES profile at time index i to predict time index i+1 using TKE
            f = closure_free_convection_tke(D, Δt, les)
            evolve_physics_model_fn2(T⁰) = f(problem.parameters; T⁰=T⁰, n_steps=1)[:,2]
            tke_data = Array{Array{Float64,1},1}(UndefInitializer(), length(les.t))
            tke_data[1] = custom_avg(v[:,1], D)
            for i in 1:length(les.t)-1
                tke_data[i+1] = evolve_physics_model_fn2(v[:,i])
            end
            return Sequential_TKE("T", tke_data, evolve_physics_model_fn2, Δt)

        else; throw(error())
        end


    elseif typeof(problem) <: ResidualProblem

        if problem.type == "KPP"
            # Use the LES profile at time index i to predict time index i+1 using KPP
            f = closure_free_convection_kpp(D, Δt, les)
            evolve_physics_model_fn3(T⁰) = f(problem.parameters; T⁰=T⁰, n_steps=1)[:,2]
            kpp_data = Array{Array{Float64,1},1}(UndefInitializer(), length(les.t))
            kpp_data[1] = custom_avg(v[:,1], D)
            for i in 1:length(les.t)-1
                kpp_data[i+1] = evolve_physics_model_fn3(v[:,i])
            end
            return Residual_KPP("T", kpp_data, evolve_physics_model_fn3, Δt)

        elseif problem.type == "TKE"
            # Use the LES profile at time index i to predict time index i+1 using TKE
            f = closure_free_convection_tke(D, Δt, les)
            evolve_physics_model_fn4(T⁰) = f(problem.parameters; T⁰=T⁰, n_steps=1)[:,2]
            tke_data = Array{Array{Float64,1},1}(UndefInitializer(), length(les.t))
            tke_data[1] = custom_avg(v[:,1], D)
            for i in 1:length(les.t)-1
                tke_data[i+1] = evolve_physics_model_fn4(v[:,i])
            end
            return Residual_TKE("T", tke_data, evolve_physics_model_fn4, Δt)

        else; throw(error())
        end

    elseif typeof(problem) <: SlackProblem

        if problem.type == "KPP"
            # Predict the full evolution of the temperature profile from the initial time step using KPP
            f = closure_free_convection_kpp_full_evolution(D, Δt, les)
            kpp_data = f(problem.parameters)
            kpp_data = [kpp_data[:,i] for i in 1:length(les.t)]
            return Slack_KPP("T", kpp_data)

        elseif problem.type == "TKE"
            # Predict the full evolution of the temperature profile from the initial time step using TKE
            f = closure_free_convection_tke_full_evolution(D, Δt, les)
            tke_data = f(problem.parameters)
            tke_data = [tke_data[:,i] for i in 1:length(les.t)]
            return Slack_TKE("T", tke_data)

        else; throw(error())
        end

    else; throw(error())
    end

end
