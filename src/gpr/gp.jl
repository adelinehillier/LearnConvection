"""
Adapted from sandreza/Learning/sandbox/gaussian_process.jl
https://github.com/sandreza/Learning/blob/master/sandbox/gaussian_process.jl
Changed handling of kernel functions; changed some variable names;
added log marginal likelihood function.
"""

using LinearAlgebra

"""
GP
# Description
- data structure for typical GPR computations
# Data Structure and Description
    kernel::ℱ, a Kernel object
    x_train::𝒮 , an array of vectors (n-length array of D-length vectors)
    α::𝒮2 , an array
    K::𝒰 , matrix or sparse matrix
    CK::𝒱, cholesky factorization of K
"""
struct GP{Kernel, 𝒮, 𝒮2, 𝒰, 𝒱}
    kernel::Kernel
    x_train::𝒮
    α::𝒮2
    K::𝒰
    CK::𝒱
end

"""
construct_gpr(x_train, y_train; kernel; sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))
# Description
Constructs the posterior distribution for a gp. In other words this does the 'training' automagically.
# Arguments
- `x_train`: (array). training inputs (predictors), must be an array of states.
                      length-n array of D-length vectors, where D is the length of each input n is the number of training points.
- `y_train`: (array). training outputs (prediction), must have the same number as x_train
                      length-n array of D-length vectors.
- `kernel`: (Kernel). Kernel object. See kernels.jl.
                      kernel_function(kernel)(x,x') maps predictor x predictor to real numbers.
# Keyword Arguments
- `z`: (vector). values w.r.t. which to derivate the state (default none).
- `normalize`: (bool). whether to normalize the data during preprocessing and reverse the scaling for postprocessing. Can lead to better performance.
- `hyperparameters`: (array). default = []. hyperparameters that enter into the kernel
- `sparsity_threshold`: (number). default = 0.0. a number between 0 and 1 that determines when to use sparse array format. The default is to never use it
- `robust`: (bool). default = true. This decides whether to uniformly scale the diagonal entries of the Kernel Matrix. This sometimes helps with Cholesky factorizations.
- `entry_threshold`: (number). default = sqrt(eps(1.0)). This decides whether an entry is "significant" or not. For typical machines this number will be about 10^(-8) * largest entry of kernel matrix.
# Return
- GP object
"""
function model(x_train, y_train, kernel::Kernel, zavg; sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))

    # get k(x,x') function from kernel object
    kernel = kernel_function(kernel; z=zavg)
    # fill kernel matrix with values
    K = compute_kernel_matrix(kernel, x_train)

    # get the maximum entry for scaling and sparsity checking
    mK = maximum(K)

    # make Cholesky factorization work by adding a small amount to the diagonal
    if robust
        K += mK*sqrt(eps(1.0))*I
    end

    # check sparsity
    bools = K .> entry_threshold * mK
    sparsity = sum(bools) / length(bools)
    if sparsity < sparsity_threshold
        sparse_K = similar(K) .* 0
        sparse_K[bools] = sK[bools]
        K = sparse(Symmetric(sparse_K))
        CK = cholesky(K)
    else
        CK = cholesky(K)
    end

    # get prediction weights FIX THIS SO THAT IT ALWAYS WORKS
    y = hcat(y_train...)'
    α = CK \ y # α = K + σ_noise*I

    # construct struct
    return GP(kernel, x_train, α, K, Array(CK))
end

function model(𝒟::ProfileData; kernel::Kernel = Kernel())
    # create instance of GP using data from ProfileData object
    𝒢 = model(𝒟.x_train, 𝒟.y_train, kernel, 𝒟.zavg);
    return 𝒢
end

"""
prediction(x, 𝒢::GP)
# Description
- Given state x, GP 𝒢, returns the mean GP prediction
# Arguments
- `x`: scaled state
- `𝒢`: GP object with which to make the prediction
# Return
- `y`: scaled prediction
"""
function model_output(x, 𝒢::GP)
    return 𝒢.α' * 𝒢.kernel.([x], 𝒢.x_train)
end

"""
uncertainty(x, 𝒢::GP)
# Description
- Given state x and GP 𝒢, output the variance at a point
# Arguments
- `x`: state
# Return
- `var`: variance
"""
function uncertainty(x, 𝒢::GP)
    tmpv = zeros(size(𝒢.x_train)[1])
    for i in eachindex(𝒢.x_train)
        tmpv[i] = 𝒢.kernel(x, 𝒢.x_train[i])
    end
    # no ldiv for suitesparse
    tmpv2 = 𝒢.CK \ tmpv
    var = k(x, x) .- tmpv'*tmpv2  # var(f*) = k(x*,x*) - tmpv'*tmpv2
    return var
end

"""
compute_kernel_matrix(kernel, x)
# Description
- Computes the kernel matrix for GPR
# Arguments
- `k` : (Kernel) kernel function k(a,b).
- `x` : (array of predictors). x[1] is a vector
# Return
- `sK`: (symmetric matrix). A symmetric matrix with entries sK[i,j] = k(x[i], x[j]). This is only meaningful if k(x,y) = k(y,x) (it should)
"""
function compute_kernel_matrix(k, x)

    K = [k(x[i], x[j]) for i in eachindex(x), j in eachindex(x)]

    if typeof(K[1,1]) <: Number
        sK = Symmetric(K)
    else
        sK = K
    end
    return sK
end

"""
mean_log_marginal_loss(y_train, 𝒢::GP; add_constant=false)
# Description
Computes log marginal loss for each element in the output and averages the results.
Assumes noise-free observations.

log(p(y|X)) = -(1/2) * (y'*α + 2*sum(Diagonal(CK)) + n*log(2*pi))
where n is the number of training points and

# Arguments
- `y_train`: (Array). training outputs (prediction), must have the same number as x_train
- `𝒢`: (GP).
# Keyword Arguments
- `add_constant`: (bool). whether to give the exact value of the loss or leave out an added constant for efficiency.

"""
function mean_log_marginal_loss(y_train, 𝒢::GP; add_constant=false)
    n = length(𝒢.x_train)
    D = length(𝒢.x_train[1])

    ys = hcat(y_train...)' # n x D

    if add_constant
        c = sum([log(𝒢.CK[i,i]) for i in 1:n]) + 0.5*n*log(2*pi)
        total_loss=0.0
        for i in 1:D
            total_loss -= 0.5*ys[:,i]'*𝒢.α[:,i] + c
        end
    else
        total_loss=0.0
        for i in 1:D
            total_loss -= 0.5*ys[:,i]'*𝒢.α[:,i]
        end
    end

    return total_loss / D
end

"""
# Description
Predict profile across all time steps for the true check.
    - if the problem is sequential, predict profiles from start to finish without the training, using only the initial profile as the initial condition.
    - if the problem is residual, predict profiles at each timestep using model-predicted difference between truth and physics-based model (KPP or TKE) prediction

Returns an n-length array of D-length vectors, where n is the number of training points and D is the
# Arguments
- `𝒢` (GP). The GP object
- `𝒟` (ProfileData). The ProfileData object whose data 𝒢 will "test on", not necessarily the object that was used to train 𝒢.
                      If the problem is sequential, 𝒟 is the ProfileData object whose starting profile will be evolved forward using 𝒢.
# Keyword Arguments
- `postprocessed` (bool or String). If false, return whatever the model predicts directly.
                                    If true, return the full predicted temperature profile calculated from the model output.
                                    If "both", return both.
"""
function predict(𝒢::GP, 𝒟::ProfileData; postprocessed=true)

    if typeof(𝒟.problem) <: SequentialProblem

        # Predict temperature profile from start to finish without the training data.
        gpr_prediction = similar(𝒟.y)
        gpr_prediction[1] = 𝒟.y[1] # starting profile

        for i in 1:(length(𝒟.y)-1)
            y_prediction = model_output(gpr_prediction[i], 𝒢)
            gpr_prediction[i+1] = y_prediction
        end

    elseif typeof(𝒟.problem) <: ResidualProblem

        # Predict temperature profile at each timestep using model-predicted difference between truth and physics-based model (KPP or TKE) prediction
        gpr_prediction = [model_output(𝒟.x[i], 𝒢) for i in 1:(𝒟.Nt)]

    else; throw(error)
    end

    if postprocessed == "both"
        return (gpr_prediction, get_postprocessed_predictions(𝒟.x, gpr_prediction, 𝒟.all_problems))
    end

    if postprocessed
        return get_postprocessed_predictions(𝒟.x, gpr_prediction, 𝒟.all_problems)
    end

    return gpr_prediction
end

function get_postprocessed_predictions(x, gpr_prediction, all_problems)

    result=Array{Array{Float64,1},1}()
    i=1

    println("length second opinion $(length(all_problems))")

    for (problem, n_x) in all_problems # n_x: number of predictors for that problem
        println(n_x)
        result = vcat(result, postprocess_prediction(x[i : i+n_x-1], gpr_prediction[i : i+n_x-1], problem))
        i += n_x
    end

    return result

end
