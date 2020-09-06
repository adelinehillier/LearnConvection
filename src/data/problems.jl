"""
Used in gp.jl.
Data pre- / post-processing. Takes a ProfileData object and prepares it for use in GP.
"""

abstract type Problem end

abstract type SequentialProblem <: Problem end
abstract type ResidualProblem <: Problem end

struct Sequential <: SequentialProblem # for mappings that predict the subsequent timestep from the current timestep
    type::String # "T" or "dT" or "wT"
end

struct Residual <: ResidualProblem # for mappings that predict the true current timestep from a physics-based model's current timestep
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
----- Description
    Creates an instance of a Problem struct depending on the type of mapping.
    Returns the problem and the correct data array from the OceananigansData objectssociated with the variable of interest.
----- Arguments
- 'problem': (SequentialProblem).  what mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"))
- 'les': (OceananigansData).       what mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"))
- 'N²': (Number)                   initial buoyancy stratification
- 'timeseries': (Array)            simulation timeseries [s]
"""
function get_problem_v(problem, les, N², D, timeseries)

    Δt = timeseries[2]-timeseries[1]

    if typeof(problem) <: SequentialProblem

        if problem.type == "dT"
            Δt /= N² # Δt' = Δt / N²
            return les.T, Sequential_dT("T", Δt) # Sequential_dT(variable, Δt)

        elseif problem.type == "T"
            return les.T, Sequential_T("T")

        elseif problem.type == "wT"
            return les.wT, Sequential_wT("wT")

        else; throw(error())
        end

    elseif typeof(problem) <: ResidualProblem

        if problem.type == "KPP"
            f = closure_free_convection_kpp(D, Δt, les)
            kpp_data = f(problem.parameters)
            kpp_data = [kpp_data[:,i] for i in 1:length(timeseries)]
            return les.T, Residual_KPP("T", kpp_data)

        elseif problem.type == "TKE"
            f = closure_free_convection_tke(D, Δt, les)
            tke_data = f(problem.parameters)
            tke_data = [tke_data[:,i] for i in 1:length(timeseries)]
            return les.T, Residual_TKE("T", tke_data)

        else; throw(error())
        end

    else; throw(error())
    end

end
