"""
Data pre- / post-processing for sequential problems. Takes a ProfileData object and prepares it for use in GP.
"""

"""
`get_predictors_targets(vavg::Array, problem::Sequential_T)`
# Description
    Returns x and y, the scaled predictor and target pairs from which to extract the training and verification data sets for "T" profiles.

    model( predictor ) -> target
         model( T[i] ) -> (T[i+1]-T[i])/Δt ≈ ∂t(T)

# Arguments
- `vavg`: (Array)                Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
- `problem`: (Sequential_dT)     Sequential_T object associated with the data (output of get_problem)

# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
# | Sequential_dT                              |
# |                                            |
# |   predictor       target                   |
# |   T[i] --model--> (T[i+1]-T[i])/Δt ≈ ∂t(T) |
# |                                            |
# |   model output                             |
# |   model(T[i])                              |
# |                                            |
# |   prediction                               |
# |   predicted T[i+1] = model(T[i])Δt + T[i]  |
# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

"""
function get_predictors_targets(vavg::Array, problem::Sequential_dT)

    # scale according to initial temperature profile vavg[1]
    scaling = min_max_scaling(vavg[1][end]-vavg[1][1], minimum(vavg[1]))
    vavg = [scale(vec, scaling) for vec in vavg]

    # store the scaling as an attribute in the problem for later use
    problem.scaling = scaling

    # calculate predictors and targets from the scaled data
    predictors = vavg[1:end-1] # T[i] for i = 1, ..., Nt-1
    targets = (vavg[2:end] - predictors) / problem.Δt # (T[i+1]-T[i])/Δt'

    return (predictors, targets)
end

"""
`postprocess_prediction(predictor, model_output, problem::Sequential_dT)`

# Description
Takes in a scaled predictor, T[i], the scaled model output on the predictor, model(T[i]), and a Sequential_T object.
Returns the unscaled prediction (predicted temperature profile), T[i+1], computed from T[i] and model(T[i]) by

    predicted T[i+1] = model(T[i])Δt + T[i]

# Arguments
- `predictor`: (Array)            T[i], the scaled predictor for a temperature profile
- `model_output`: (Array)         model(T[i]), the scaled model output for a temperature profile
- `problem`: (Sequential_dT)
"""
function postprocess_prediction(predictor, model_output, problem::Sequential_dT)

    prediction = model_output * problem.Δt + predictor
    prediction = [unscale(vec, problem.scaling) for vec in prediction]

    return prediction

end

"""
`get_predictors_targets(vavg::Array, problem::Sequential_T)`

# Description
    Returns x and y, the predictors and target predictions from which to extract the training and verification data for "T" profiles.

    model( predictors ) -> targets
          model( T[i] ) -> T[i+1]

# Arguments
- `vavg`: (Array)               Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
- `problem`: (Sequential_T)     Sequential_T object associated with the data (output of get_problem)

# *--*--*--*--*--*--*--*--*--*--*
# | Sequential_T                |
# |   T[i] --model--> T[i+1]    |
# *--*--*--*--*--*--*--*--*--*--*
"""
function get_predictors_targets(vavg, problem::Sequential_T)

    # scale according to initial temperature profile vavg[1]
    scaling = min_max_scaling(vavg[1][end]-vavg[1][1], minimum(vavg[1]))
    vavg = [scale(vec, scaling) for vec in vavg]

    # store the scaling as an attribute in the problem for later use
    problem.scaling = scaling # this mutates the input

    # get predictors and targets
    predictors = vavg[1:end-1] # T[i] for i = 1, ..., Nt-1
    targets = vavg[2:end] # T[i+1]

    return (predictors, targets)
end

"""
`postprocess_prediction(predictor, prediction, problem::Sequential_T)`

# Description
Takes in a scaled predictor, T[i], the scaled GP prediction on the predictor, G(T[i]), and a Sequential_T object.
Returns the predicted temperature profile, T[i+1], computed from T[i] and G(T[i]) by

           prediction = model( predictor )
     predicted T[i+1] = model( T[i] )

# Arguments
- `predictor`: (Array)            T[i], the scaled predictor for a temperature profile
- `prediction`: (Array)           model(T[i), the scaled prediction for a temperature profile
- `problem`: (Sequential_T)
"""
function postprocess_prediction(predictor, prediction, problem::Sequential_T)
    # println("prediction$([unscale(vec, problem.scaling) for vec in prediction])")

    return [unscale(vec, problem.scaling) for vec in prediction]
end

"""
`get_predictors_targets(vavg, problem::Sequential_wT)`

# Description
    Returns x and y, the predictors and targets from which to extract the training and verification data for "wT" profiles.

     model( predictor ) -> target
         model( wT[i] ) -> wT[i+1]

# Arguments
- `vavg`: (Array)                  Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
- `problem`: (Sequential_wT)       Sequential_wT object associated with the data (output of get_problem)

# *--*--*--*--*--*--*--*--*--*--*
# | Sequential_wT               |
# |   wT[i] --model--> wT[i+1]  |
# *--*--*--*--*--*--*--*--*--*--*
"""
function get_predictors_targets(vavg, problem::Sequential_wT)

    # scale according to initial temperature profile vavg[1]
    # scaling = min_max_scaling(vavg[1][end]-vavg[1][1], minimum(vavg[1]))
    # vavg = [scale(vec, scaling) for vec in vavg]

    # store the scaling as an attribute in the problem for later use
    # problem.scaling = scaling # this mutates the input

    # get predictors and predictions
    predictors = vavg[1:end-1] # wT[i] for i = 1, ..., Nt-1
    targets = vavg[2:end] # wT[i+1]

    return (predictors, targets)
end

"""
`postprocess_prediction(predictor, prediction, problem::Sequential_wT)`

# Description
Takes in a scaled predictor, wT[i], the scaled model prediction on the predictor, model(wT[i]), and a Sequential_wT object.
Returns the temperature profile, T[i+1], computed from model(T[i]) by

           prediction = model(predictor)
    predicted wT[i+1] = model(wT[i])

# Arguments
- `predictor`: (Array)                wT[i], the predictor for a wT profile
- `prediction`: (Array)               predicted wT[i+1], the model prediction for a wT profile
- `problem`: (Sequential_wT)          Sequential_wT object associated with the data
"""
function postprocess_prediction(predictor, prediction, problem::Sequential_wT)
    # prediction = [unscale(vec, problem.scaling) for vec in prediction]
    return prediction
end

"""
get_predictors_targets(vavg::Array, problem::Residual_KPP)
# Description
    Returns x and y, the scaled predictors and target predictions from which to extract the training and verification data for temperature profiles.
    Scales the predictors and targets using min-max scaling based on the initial temperature profile from the les simulation.

    model( predictors ) = targets
    model( KPP(T[i])  ) = T[i] - KPP(T[i])

# Arguments
- `vavg`: (Array)               Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
-  `problem`: (Residual_KPP)    Residual_KPP object associated with the data (output of get_problem)

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
function get_predictors_targets(vavg::Array, problem::Union{Sequential_KPP, Sequential_TKE})

    # scale according to initial temperature profile vavg[1]
    scaling = min_max_scaling(vavg[1][end]-vavg[1][1], minimum(vavg[1]))
    vavg = [scale(vec, scaling) for vec in vavg]

    # store the scaling as an attribute in the problem for later use
    problem.scaling = scaling

    # scale physics data with same scaling
    physics_data = [scale(vec, scaling) for vec in problem.physics_data] # kpp(i; T[i-1])

    targets = (vavg .- physics_data) / problem.Δt # residual
    return (vavg, targets)
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
function postprocess_prediction(predictor, prediction, problem::Union{Sequential_KPP, Sequential_TKE})

    #unscale predictor
    predictor = [unscale(vec, problem.scaling) for vec in predictor] # kpp data

    #unscale prediction
    prediction = [(vec * problem.scaling.Δv) for vec in prediction] # residual

    return predictor .+ prediction * problem.Δt
end
