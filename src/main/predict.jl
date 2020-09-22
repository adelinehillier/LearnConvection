"""
# Description
Predict profile across all time steps for the true check.
    - if the problem is sequential, predict profiles from start to finish without the training, using only the initial profile as the initial condition.
    - if the problem is residual, predict profiles at each timestep using model-predicted difference between truth and physics-based model (KPP or TKE) prediction

Returns an n-length array of D-length vectors, where n is the number of training points and D is the
# Arguments
- `ℳ` (GP or NN). The model you will use to approximate the mapping.
- `𝒟` (ProfileData). The ProfileData object whose data 𝒢 will "test on", not necessarily the object that was used to train 𝒢.
                      If the problem is sequential, 𝒟 is the ProfileData object whose starting profile will be evolved forward using 𝒢.
# Keyword Arguments
- `postprocessed` (bool or String). If false, return whatever the model predicts directly (direct model output)
                                    If true, return the full predicted temperature profile calculated from the model output.
                                    If "both", return both.
"""

function predict(ℳ, 𝒟::ProfileData; postprocessed=true)

    if typeof(𝒟.problem) <: Union{Sequential_KPP, Sequential_TKE}
        # Predict temperature profile from start to finish without the training data.

        gpr_prediction=Array{Array{Float64,1},1}()
        postprocessed_prediction=Array{Array{Float64,1},1}()
        i=1
        for (problem, n_x) in 𝒟.all_problems # n_x: number of predictors (i.e. time steps) for that problem

            post_pred_chunk = Array{Array{Float64,1},1}(UndefInitializer(), n_x)
            gpr__pred_chunk = Array{Array{Float64,1},1}(UndefInitializer(), n_x)
            post_pred_chunk[1] = unscale(𝒟.x[i], problem.scaling) # should = unscale(𝒟.x[i], problem.scaling) = postprocess_prediction(𝒟.x[i], 𝒟.y[i], problem)
            gpr__pred_chunk[1] = 𝒟.y[i] # residual -- should be zeros for this initial time step. Good sanity check.

            for i in 1:n_x-1
                kpp_pred = scale(problem.evolve_physics_model_fn(post_pred_chunk[i]), problem.scaling)
                gpr__pred_chunk[i+1] = model_output(scale(post_pred_chunk[i], problem.scaling), ℳ)
                post_pred_chunk[i+1] = postprocess_prediction(kpp_pred, gpr__pred_chunk[i+1], problem)
            end

            gpr_prediction = vcat(gpr_prediction, gpr__pred_chunk)
            postprocessed_prediction = vcat(postprocessed_prediction, post_pred_chunk)
            i += n_x
        end

    elseif typeof(𝒟.problem) <: ResidualProblem
        # Predict temperature profile from start to finish without the training data.

        gpr_prediction=Array{Array{Float64,1},1}()
        postprocessed_prediction=Array{Array{Float64,1},1}()
        i=1
        for (problem, n_x) in 𝒟.all_problems # n_x: number of predictors (i.e. time steps) for that problem

            post_pred_chunk = Array{Array{Float64,1},1}(UndefInitializer(), n_x)
            gpr__pred_chunk = Array{Array{Float64,1},1}(UndefInitializer(), n_x)
            post_pred_chunk[1] = unscale(𝒟.x[i], problem.scaling) # should = unscale(𝒟.x[i], problem.scaling) = postprocess_prediction(𝒟.x[i], 𝒟.y[i], problem)
            gpr__pred_chunk[1] = 𝒟.y[i] # residual -- should be zeros for this initial time step. Good sanity check.

            for i in 1:n_x-1
                kpp_pred = scale(problem.evolve_physics_model_fn(post_pred_chunk[i]), problem.scaling)
                gpr__pred_chunk[i+1] = model_output(kpp_pred, ℳ)
                post_pred_chunk[i+1] = postprocess_prediction(kpp_pred, gpr__pred_chunk[i+1], problem)
            end

            gpr_prediction = vcat(gpr_prediction, gpr__pred_chunk)
            postprocessed_prediction = vcat(postprocessed_prediction, post_pred_chunk)
            i += n_x
        end

    elseif typeof(𝒟.problem) <: SequentialProblem
        # Predict temperature profile from start to finish without the training data.
        gpr_prediction = similar(𝒟.y)
        gpr_prediction[1] = 𝒟.y[1] # starting profile

        for i in 1:(length(𝒟.y)-1)
            y_prediction = ℳ_output(gpr_prediction[i], ℳ)
            gpr_prediction[i+1] = y_prediction
        end
        postprocessed_prediction = get_postprocessed_predictions(𝒟.x, gpr_prediction, 𝒟.all_problems)

    elseif typeof(𝒟.problem) <: SlackProblem
        # Predict temperature profile at each timestep using model-predicted difference between truth and physics-based model (KPP or TKE) prediction
        gpr_prediction = [GaussianProcess.model_output(𝒟.x[i], ℳ) for i in 1:(𝒟.Nt)]
        postprocessed_prediction = get_postprocessed_predictions(𝒟.x, gpr_prediction, 𝒟.all_problems)
        ##
        # animate_physics_profile(ℳ, 𝒟)

    else; throw(error)
    end

    if postprocessed == "both";
        return (gpr_prediction, postprocessed_prediction)
    end

    if postprocessed
        return postprocessed_prediction
    end

    return gpr_prediction
end

function get_postprocessed_predictions(x, gpr_prediction, all_problems)

    result=Array{Array{Float64,1},1}()
    i=1
    for (problem, n_x) in all_problems # n_x: number of predictors for that problem
        result = vcat(result, GaussianProcess.postprocess_prediction(x[i : i+n_x-1], gpr_prediction[i : i+n_x-1], problem))
        i += n_x
    end

    return result
end

# function animate_physics_profile(ℳ, 𝒟)
#
#     variable = 𝒟.problem.variable # "T" or "wT"
#     xlims = x_lims[variable]
#
#     physics_data=Array{Array{Float64,1},1}()
#     for (problem, n_x) in 𝒟.all_problems # n_x: number of predictors for that problem
#         physics_data = vcat(physics_data, problem.physics_data)
#     end
#
#     # Compute error
#     n = length(physics_data)
#     total_error = 0.0
#     for i in 1:n
#         total_error += euclidean_distance(𝒟.vavg[i], physics_data[i])
#     end
#     println("ERROR $(total_error / n)")
#     println(total_error / n)
#
#     # Animate
#     animation_set = 1:30:(length(physics_data)-2)
#     anim = @animate for i in animation_set
#         day_string = string(floor(Int, 𝒟.t[i]/86400))
#         scatter(physics_data[i], 𝒟.zavg, label = "Physics") ####
#         plot!(𝒟.v[:,i], 𝒟.z, legend = :topleft, label = "LES", xlabel = "$(long_name[variable])", ylabel = "Depth [m]", title = "day " * day_string, xlims=xlims)
#     end
#
#     gif(anim, "physics_plot.gif")
# end
