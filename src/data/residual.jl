"""
Data pre- / post-processing for residual problems. Takes a ProfileData object and prepares it for use in GP.
"""

"""
get_predictors_targets(vavg::Array, problem::Union{Residual_KPP, Residual_TKE})
# Description
    Returns x and y, the scaled predictors and target predictions from which to extract the training and verification data for temperature profiles.
    Scales the predictors and targets using min-max scaling based on the initial temperature profile from the les simulation.

    model( predictors ) = targets
    model( KPP(T[i])  ) = T[i] - KPP(T[i])

# Arguments
- `vavg`: (Array)      Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
-  `problem`: (Union{Residual_KPP, Residual_TKE})    Residual_KPP object associated with the data (output of get_problem)

# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
# | Residual_KPP                                     |
# |                                                  |
# |   predictor            target                    |
# |   KPP(T[i]) --model--> T[i] - KPP(T[i])          |
# |                                                  |
# |   model output                                   |
# |   model(KPP(T[i]))                               |
# |                                                  |
# |   predicted T[i] = model(KPP(T[i])) + KPP(T[i])  |
# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
# | Residual_TKE                                     |
# |                                                  |
# |   predictor            target                    |
# |   TKE(T[i]) --model--> T[i] - TKE(T[i])          |
# |                                                  |
# |   model output                                   |
# |   model(TKE(T[i]))                               |
# |                                                  |
# |   predicted T[i] = model(TKE(T[i])) + TKE(T[i])  |
# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

"""
function get_predictors_targets(vavg::Array, problem::Union{Residual_KPP, Residual_TKE})

    # scale according to initial temperature profile vavg[1]
    scaling = min_max_scaling(vavg[1][end]-vavg[1][1], minimum(vavg[1]))
    vavg = [scale(vec, scaling) for vec in vavg]

    # store the scaling as an attribute in the problem for later use
    problem.scaling = scaling

    # scale kpp data with same scaling
    predictors = [scale(vec, scaling) for vec in problem.physics_data] # kpp(i; T[i-1]) or # tke(i; T[i-1])

    targets = (vavg .- predictors) / problem.Δt # residual
    return (predictors, targets)
end

"""
# Description
Takes in a predictor, T[i], the GP prediction on the predictor, G(T[i]), and a Residual_KPP object.
Returns the predicted temperature profile, T[i+1], computed from T[i] and G(T[i]) by

           prediction = model(predictor) # residual
     predicted T[i+1] = model(KPP(T[i])) + KPP(T[i])

# Arguments
- `predictor`: (Array)            T[i], the predictor for a temperature profile
- `prediction`: (Array)           model(T[i), the prediction for a temperature profile
- `problem`: (Residual_KPP)
"""
function postprocess_prediction(predictor, prediction, problem::Union{Residual_KPP, Residual_TKE})

    #unscale predictor
    predictor = [unscale(vec, problem.scaling) for vec in predictor] # kpp data

    #unscale prediction
    prediction = [(vec * problem.scaling.Δv) for vec in prediction] # residual

    return predictor .+ prediction * problem.Δt
end
