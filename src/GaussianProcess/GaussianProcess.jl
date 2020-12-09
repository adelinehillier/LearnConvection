"""
Includes all useful functions for applying GPR to T and wT profiles from Oceananigans.jl simulations.
Uses ProfileData struct to store data and GP struct for performing GPR on the data in ProfileData object.
"""
module GaussianProcess

using LearnConvection.Data

include("kernels.jl")
export  Kernel,
        SquaredExponentialI,
        RationalQuadraticI,
        Matern12I,
        Matern32I,
        Matern52I,
        SpectralMixtureProductI,
        SpectralMixtureProductA,
        kernel_function

include("distances.jl")
export  euclidean_distance,
        derivative_distance,
        antiderivative_distance

include("gp.jl")
export  model_output,
        uncertainty,
        compute_kernel_matrix,
        mean_log_marginal_loss
export  model
        # predict

export get_kernel

"""
```get_kernel(kernel_id::Int64, logγ, logσ, distance; logα=0.0)```

# Description
Returns a Kernel object with the specified parameters.

# Arguments
- `kernel_id`: (Int64). Identification number for the kernel type (see kernel options)
- `logγ`: (Float64). Log(length scale) parameter.
- `logσ`: (Float64). Log(signal variance) parameter.
- `distance`: (Function). Distance function to use in the kernel.

# Keyword Arguments
- `logα`: (Float64). Log(α) parameter if kernel_id = 5.

# kernel options
 1   =>   "Squared exponential kernel:        k(x,x') = σ * exp( - d(x,x')² / 2γ² )",
 2   =>   "Matérn with ʋ=1/2:                 k(x,x') = σ * exp( - d(x,x') / γ )",
 3   =>   "Matérn with ʋ=3/2:                 k(x,x') = σ * (1+c) * exp(-√(3)*d(x,x'))/γ)",
 4   =>   "Matérn with ʋ=5/2:                 k(x,x') = σ * ( 1 + √(5)*d(x,x'))/γ + 5*d(x,x')²/(3*γ^2) ) * exp(-√(5)*d(x,x'))/γ)",
 5   =>   "Rational quadratic kernel:         k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",
"""
# function get_kernel(kernel_id::Int64, logγ, logσ, distance; logα=0.0)
#         # convert from log10 scale
#         γ = 10^logγ
#         σ = 10^logσ
#         α = 10^logα
#         if kernel_id==1; return SquaredExponentialI(γ, σ, distance) end
#         if kernel_id==2; return Matern12I(γ, σ, distance) end
#         if kernel_id==3; return Matern32I(γ, σ, distance) end
#         if kernel_id==4; return Matern52I(γ, σ, distance) end
#         if kernel_id==5; return RationalQuadraticI(γ, σ, α, distance)
#         else; throw(error()) end
# end

function get_kernel(kernel_id::Int64, args...)
        # args = f.(args) # convert from log10 scale

        function g(x)
                if typeof(x) <: Number
                        return 10^x end
                return x
        end

        args = g.(args)

        if kernel_id==1; return SquaredExponentialI(args...) end
        if kernel_id==2; return Matern12I(args...) end
        if kernel_id==3; return Matern32I(args...) end
        if kernel_id==4; return Matern52I(args...) end
        if kernel_id==5; return RationalQuadraticI(args...) end
        if kernel_id==6; return SpectralMixtureProductI(args...)
        else; throw(error()) end
end

end
