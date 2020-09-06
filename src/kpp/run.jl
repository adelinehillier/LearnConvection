"""
Adapted from
https://github.com/sandreza/OceanConvectionUQSupplementaryMaterials/blob/master/src/ForwardMap/fm.jl
updated for latest version of OceanTurb
"""

# include("../les/custom_avg.jl")
# include("../les/output_to_data.jl")
# include("../les/get_les_data.jl")
# include("../data/scalings.jl")

# using OceanTurb

"""
closure_free_convection(N, Δt, les::OceananigansData; subsample = 1, grid = 1)
# Description
- constructs forward map. Assumes initial conditions and boundary conditions are taken from les data.
# Arguments
- `N`: number of gridpoints to output to
- `Δt`: time step size in seconds
- `les`: les data of the OceananigansData type
# Keyword Arguments
- `subsample`: indices to subsample in time,
- `grid`: in case one wants to save the model grid
# Output
- The forward map. A function that takes parameters and outputs temperature profiles
-   `𝑪`: parameters in KPP, assumes that \n
    𝑪[1]: Surface Layer Fraction  \n
    𝑪[2]: Nonlocal Flux Amplitude \n
    𝑪[3]: Diffusivity Amplitude \n
    𝑪[4]: Shear Constant \n
"""
function closure_free_convection_kpp(D, Δt, les::OceananigansData;
                                     subsample = 1, grid = 1)
    # define the closure
    function free_convection(parameters)
        # # set parameters
        # parameters = KPP.Parameters( CSL = 𝑪[1], CNL = 𝑪[2], Cb_T = 𝑪[3], CKE = 𝑪[4])
        # Build the model with a Backward Euler timestepper
        constants = Constants(Float64; α = les.α , β = les.β, ρ₀= les.ρ, cP=les.cᵖ, f=les.f⁰, g=les.g)
        model = KPP.Model(N=D, H=les.L, stepper=:BackwardEuler, constants = constants, parameters = parameters)
        # Get grid if necessary
        if grid != 1
            zp = collect(model.grid.zc)
            @. grid  = zp
        end
        # get average of initial condition of LES
        T⁰ = custom_avg(les.T⁰, D)
        # set equal to initial condition of parameterization
        model.solution.T[1:D] = copy(T⁰)
        # Set boundary conditions
        model.bcs.T.top = FluxBoundaryCondition(les.top_T)
        model.bcs.T.bottom = GradientBoundaryCondition(les.bottom_T)
        # set aside memory
        if subsample != 1
            time_index = subsample
        else
            time_index = 1:length(les.t)
        end
        Nt = length(les.t[time_index])
        𝒢 = zeros(D, Nt)

        # loop the model
        ti = collect(time_index)
        for i in 1:Nt
            t = les.t[ti[i]]
            run_until!(model, Δt, t)
            @. 𝒢[:,i] = model.solution.T[1:D]
        end
        return 𝒢
    end
    return free_convection
end


"""
closure_free_convection_flexible(N, Δt, les::OceananigansData; subsample = 1, grid = 1)
# Description
- constructs forward map. Assumes initial conditions and boundary conditions are taken from les data. Adds two more parameters compared to previous one
# Arguments
- `N`: number of gridpoints to output to
- `Δt`: time step size in seconds
- `les`: les data of the OceananigansData type
# Keyword Arguments
- `subsample`: indices to subsample in time,
- `grid`: in case one wants to save the model grid
- `power`: a method of modifying the scaling law, default is 1
# Output
- The forward map. A function that takes parameters and outputs temperature profiles
-   `𝑪`: parameters in KPP, assumes that \n
    𝑪[1]: Surface Layer Fraction  \n
    𝑪[2]: Nonlocal Flux Amplitude \n
    𝑪[3]: Diffusivity Amplitude \n
    𝑪[4]: Shear Constant, w1 * w2 \n
    𝑪[5] = Shear Constant 2
    𝑪[6]: Shear Constant 3,\n
- The extra parameters correspond to  C[5] * w1^(1-power) * w2^(1+power)
- where w1 = max(0, h * Qb)^(1/3), w2 = sqrt(𝑪[6] * h^2), are two velocity scales.
"""
function closure_free_convection_flexible(D, Δt, les::OceananigansData;
                                 subsample = 1, grid = 1, power = 0.0)
    # define the closure
    function free_convection(𝑪)
        # set parameters
        parameters = KPP.Parameters( CSL = 𝑪[1], CNL = 𝑪[2], Cb_T = 𝑪[3], CKE = 𝑪[4], CKE2 = 𝑪[5], CKE3 = 𝑪[6], CKE4 = power)
        # Build the model with a Backward Euler timestepper
        constants = Constants(Float64; α = les.α , β = les.β, ρ₀= les.ρ, cP=les.cᵖ, f=les.f⁰, g=les.g)
        model = KPP.Model(N=D, H=les.L, stepper=:BackwardEuler, constants = constants, parameters = parameters)
        # Get grid if necessary
        if grid != 1
            zp = collect(model.grid.zc)
            @. grid  = zp
        end
        # get average of initial condition of LES
        T⁰ = avg(les.T⁰, N)
        # set equal to initial condition of parameterization
        model.solution.T[1:N] = copy(T⁰)
        # Set boundary conditions
        model.bcs.T.top = FluxBoundaryCondition(les.top_T)
        model.bcs.T.bottom = GradientBoundaryCondition(les.bottom_T)
        # set aside memory
        if subsample != 1
            time_index = subsample
        else
            time_index = 1:length(les.t)
        end
        Nt = length(les.t[time_index])
        𝒢 = zeros(N, Nt)

        # loop the model
        ti = collect(time_index)
        for i in 1:Nt
            t = les.t[ti[i]]
            run_until!(model, Δt, t)
            @. 𝒢[:,i] = model.solution.T[1:N]
        end
        return 𝒢
    end
    return free_convection
end
