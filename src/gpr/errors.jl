# include("gp.jl")
# include("distances.jl")

function get_me_true_check(𝒢::GP, 𝒟::ProfileData)
    # mean error on true check for a single value of γ

    total_error = 0.0
    gpr_prediction = predict(𝒢, 𝒟; postprocessed=true)
    # println("Nt-1$(𝒟.Nt-1)")
    # println("pred$(length(gpr_prediction))")
    # println("vavg$(length(𝒟.vavg))")

    # n = 𝒟.Nt-1
    n = length(gpr_prediction)
    for i in 1:n
        exact    = 𝒟.vavg[i]
        predi    = gpr_prediction[i]
        total_error += euclidean_distance(exact, predi) # euclidean distance
    end

    return total_error / n
end

function get_me_greedy_check(𝒢::GP, 𝒟::ProfileData)
    # mean error on greedy check
    # compares the direct model output to the target for all of the timesteps in the validation set.

    total_error = 0.0
    validation_set = 𝒟.validation_set
    n = length(validation_set)
    for j in 1:n
        test_index = validation_set[j]
        y_prediction = model_output(𝒟.x[test_index], 𝒢)
        error = euclidean_distance(y_prediction, 𝒟.y[test_index])
        total_error += error
    end
    return total_error / n
end
