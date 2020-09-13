"""
Used in gp.jl. Data pre- / post-processing. Takes a ProfileData object and prepares it for use in GP.
"""

abstract type Problem end

abstract type SequentialProblem <: Problem end
abstract type ResidualProblem <: Problem end

struct Sequential <: SequentialProblem # for mappings that predict the subsequent timestep from the current timestep
    type::String # "T" or "dT" or "wT"
end

struct Residual <: ResidualProblem # for mappings that predict the true current timestep using a physics-based model's prediction for that timestep
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

mutable struct Sequential_KPP <: ResidualProblem
    variable::String
    physics_data::Array
    evolve_physics_model_fn::Function
    scaling::Scaling
    Residual_KPP(variable, physics_data, evolve_physics_model_fn) = new(variable, physics_data, evolve_physics_model_fn) # incomplete initialization
end

mutable struct Sequential_TKE <: ResidualProblem
    variable::String
    physics_data::Array
    evolve_physics_model_fn::Function
    scaling::Scaling
    Residual_TKE(variable, physics_data, evolve_physics_model_fn) = new(variable, physics_data, evolve_physics_model_fn) # incomplete initialization
end

mutable struct Residual_KPP <: ResidualProblem
    variable::String
    physics_data::Array
    scaling::Scaling
    Residual_KPP(variable, physics_data) = new(variable, physics_data) # incomplete initialization
end

mutable struct Residual_TKE <: ResidualProblem
    variable::String
    physics_data::Array
    scaling::Scaling
    Residual_TKE(variable, physics_data) = new(variable, physics_data) # incomplete initialization
end

"""
get_problem(problem::Problem, data::OceananigansData, timeseries)
# Description
    Creates an instance of a Problem struct depending on the type of mapping.
    Returns the problem and the correct data array from the OceananigansData objectssociated with the variable of interest.
# Arguments
- 'problem': (SequentialProblem).  what mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"))
- 'les': (OceananigansData).       see src/les/get_les_data.jl
- 'N²': (Number)                   initial buoyancy stratification
"""
function get_problem_v(problem, les, N², D)

    Δt = les.t[2]-les.t[1]

    if typeof(problem) <: SequentialProblem

        if problem.type == "dT"
            Δt /= N² # Δt' = Δt / N²
            return les.T, Sequential_dT("T", Δt) # Sequential_dT(variable, Δt)

        elseif problem.type == "T"
            return les.T, Sequential_T("T")

        elseif problem.type == "wT"
            return les.wT, Sequential_wT("wT")

        elseif problem.type == "KPP"
            # Use the LES profile at time index i to predict time index i+1 using KPP
            f = closure_free_convection_kpp(D, Δt, les)
            evolve_physics_model_fn(T⁰) = f(problem.parameters; T⁰=T⁰, n_steps=1)[:,2]
            kpp_data = zeros(length(les.t))
            kpp_data[1] = les.T[:,1]
            for i in 1:length(les.t)-1
                kpp_data[i+1] = evolve_physics_model_fn(les.T[:,i])
            end
            return les.T, Sequential_KPP("T", kpp_data, evolve_physics_model_fn)

        elseif problem.type == "TKE"
            # Use the LES profile at time index i to predict time index i+1 using TKE
            f = closure_free_convection_tke(D, Δt, les)
            evolve_physics_model_fn(T⁰) = f(problem.parameters; T⁰=T⁰, n_steps=1)[:,2]
            tke_data = zeros(length(les.t))
            tke_data[1] = les.T[:,1]
            for i in 1:length(les.t)-1
                tke_data[i+1] = evolve_physics_model_fn(les.T[:,i])
            end
            return les.T, Sequential_TKE("T", tke_data, evolve_physics_model_fn)

        else; throw(error())
        end

    elseif typeof(problem) <: ResidualProblem

        if problem.type == "KPP"
            # Predict the full evolution of the temperature profile from the initial time step using KPP
            f = closure_free_convection_kpp_full_evolution(D, Δt, les)
            kpp_data = f(problem.parameters)
            kpp_data = [kpp_data[:,i] for i in 1:length(les.t)]
            return les.T, Residual_KPP("T", kpp_data)

        elseif problem.type == "TKE"
            # Predict the full evolution of the temperature profile from the initial time step using TKE
            f = closure_free_convection_tke_full_evolution(D, Δt, les)
            tke_data = f(problem.parameters)
            tke_data = [tke_data[:,i] for i in 1:length(les.t)]
            return les.T, Residual_TKE("T", tke_data)

        else; throw(error())
        end

    else; throw(error())
    end

end
