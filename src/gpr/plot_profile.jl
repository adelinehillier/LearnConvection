long_name = Dict("T" =>"Temperature [°C]", "wT"=>"Temperature flux [°C⋅m/s]")
x_lims = Dict("T" =>(18,20), "wT"=>(-1e-5,4e-5))

"""
plot_profile(gp::GP, data::ProfileData, V_name, time_index, gpr_prediction)
----- Description
Used in interact.jl
Plots the simulation profile at a single given index in the data timeseries along with the
corresponding GP prediction (prediciting on the training data). i.e. produces a snapshot of the profile evolution.
The gpr_prediction is computed outside this function for efficiency.
----- Arguments
- '𝒢' (GP). The GP object
- '𝒟' (ProfileData). The ProfileData object used for training and testing.
- 'V_name' (ProfileData). The ProfileData object whose starting profile will be evolved forward using 𝒢.
- 'time_index' (Int). The time index
- 'gpr_prediction' (Array). Output of get_gpr_pred (which should only be computed once) on 𝒢 and 𝒟.
"""
function plot_profile(𝒢::GP, 𝒟::ProfileData, time_index, gpr_prediction)

    day_string = string(floor(Int, 𝒟.t[time_index]/86400))
    variable = 𝒟.problem.variable # "T" or "wT"
    xlims = x_lims[variable]

    exact = 𝒟.v[:,time_index]

    p = scatter(gpr_prediction[time_index], 𝒟.zavg, label = "GP", xlims=xlims)
    plot!(exact, 𝒟.z, legend = :topleft, label = "LES", xlabel = "$(long_name[variable])", ylabel = "Depth [m]", title = "day " * day_string, xlims=xlims)
    return p
end

function animate_profile(𝒢, 𝒟)

    variable = 𝒟.problem.variable # "T" or "wT"
    xlims = x_lims[variable]

    predi = predict(𝒢, 𝒟; postprocessed=true)

    animation_set = 1:30:(𝒟.Nt-2)
    anim = @animate for i in animation_set
        exact = 𝒟.v[:,i]
        day_string = string(floor(Int, 𝒟.t[i]/86400))
        scatter(predi[i], 𝒟.zavg, label = "GP")
        plot!(exact, 𝒟.z, legend = :topleft, label = "LES", xlabel = "$(long_name[variable])", ylabel = "Depth [m]", title = "day " * day_string, xlims=xlims)
    end

    return anim
end


# Instead of plotting the full profile computed from the model output, returns the model output directly instead.
# For example, if the problem is a residual problem, plots the residual predicted by the model, not the profile computed from the residual.
function plot_model_output(𝒢, 𝒟, time_index, model_outputs)

    day_string = string(floor(Int, 𝒟.t[time_index]/86400))
    variable = 𝒟.problem.variable # "T" or "wT"
    xlims = (minimum(minimum(𝒟.y))-0.005,maximum(maximum(𝒟.y))+0.02)

    exact = 𝒟.y[time_index]

    p = scatter(model_outputs[time_index], 𝒟.zavg, label = "GP", xlims=xlims)
    plot!(exact, 𝒟.zavg, legend = false, label = "LES", xlabel = "$(long_name[variable])", ylabel = "Depth [m]", title = "Direct model output", xlims=xlims)
    return p
end

# v = 𝒟.v
# v = [v[:,i] for i in size(v)[2]]
# _ , exact_y = get_predictors_targets(v, 𝒟.problem) # what the model output should be exactly.
function animate_profile_and_model_output(𝒢, 𝒟)

    variable = 𝒟.problem.variable # "T" or "wT"
    xlims1 = x_lims[variable]
    xlims2 = (minimum(minimum(𝒟.y))-0.005, maximum(maximum(𝒟.y))+0.02)
    # xlims2 = (minimum(minimum(𝒟.y)), maximum(maximum(𝒟.y)))

    model_output, predi = predict(𝒢, 𝒟; postprocessed="both")

    ###
    # f = closure_free_convection_kpp(length(𝒟.z), 𝒟.t[2]-𝒟.t[1], get_les_data("general_strat_32_profiles.jld2")) # full resolution KPP prediction
    # kpp_data = f(OceanTurb.KPP.Parameters())
    # scaling = 𝒟.problem.scaling
    # v = scale(𝒟.v, scaling)
    # kpp_data = scale(kpp_data, scaling) # kpp(T[i])
    # target2 = v .- kpp_data
    ###

    animation_set = 1:30:(𝒟.Nt-2)
    anim = @animate for i in animation_set
        day_string = string(floor(Int, 𝒟.t[i]/86400))

        exact = 𝒟.v[:,i]
        p1 = scatter(predi[i], 𝒟.zavg, label = "GP");
        plot!(exact, 𝒟.z, legend = :topleft, label = "LES", xlabel = "$(long_name[variable])", ylabel = "Depth [m]", title = "day " * day_string, xlims=xlims1);

        # exact = target2[:,i]
        exact = 𝒟.y[i]
        p2 = scatter(model_output[i], 𝒟.zavg);
        plot!(exact, 𝒟.zavg, legend = false, xlabel = "$(long_name[variable])", ylabel = "Depth [m]", title = "Direct model output", xlims=xlims2);

        plot(p1, p2, layout=(@layout [a b]), size=(1000,500))
    end

    return anim
end
