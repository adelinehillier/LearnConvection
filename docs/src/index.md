```@meta
CurrentModule = LearnConvection
```

# LearnConvection.jl

```@index
```

```@autodocs
Modules = [LearnConvection]
```

```@docs
data(filename::String, problem::Problem; D=16, N=4)
```

### Simulation data

`src/les` Large Eddy Simulations from [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)

* Run Oceananigans LES simulations using `src/les/run.jl`

* Harvest data from the output files in `src/les/data` using `get_les_data.jl`

### Physics-based parameterizations

`src/kpp` K-Profile Parameterizations from [OceanTurb.jl](https://github.com/glwagner/OceanTurb.jl)

`src/tke` Turbulent-Kinetic-Energy-based Parameterizations from [OceanTurb.jl](https://github.com/glwagner/OceanTurb.jl)

### Machine learning-based parameterizations

`src/gpr` Gaussian Process Regression.

* Applying GPR to predict the evolution of horizontally averaged temperature or temperature flux profiles from LES simulations.

`src/nn` Neural Networks.

* Applying neural networks to predict the evolution of horizontally averaged temperature profiles from LES simulations.

***

### Data

For each set of data that we plan to train, validate, or test on, all of the relevant information goes into a `ProfileData` object. This object is handles all pre- and post-processing of the data. The relevant information is as follows.

```julia
filename = "general_strat_32_profiles.jld2"   # or filenames = [vector of filenames] to merge multiple simulations' data
problem  = Sequential("T")                    # see Problems section below
D        = 32                                 # collapse profile data down to 16 gridpoints
N        = 4                                  # collect every 4 timesteps' data for training
```

The `ProfileData` object is constructed using the `data` function (or manually: see `src/data/Data.jl` where the ProfileData struct is defined).

```julia
𝒟 = LearnConvection.Data.data(filename, problem; D=D, N=N)
```

The `ProfileData` objects used for training, validation, and testing should be created independently.

### Problems

The problem specifies which mapping we are interested in, and therefore how the data should be pre- and post-processed for the model. All problem structs are implemented in `src/gpr/problems.jl` and the scaling functions in `src/gpr/scalings.jl`

| Problem | Mapping  |       |      |
| :---    | ---:     | :---: | :--- |
| `Sequential("T")`    | ``T[i]``  | ``\xrightarrow{\text{model}} `` | ``T[i + 1] `` |
| `Sequential("dT")`   | ``T[i]``  | ``\xrightarrow{\text{model}} `` | ``(T[i+1]-T[i])/ \Delta{t'} `` |
| `Sequential("wT")`   | ``wT[i]`` | ``\xrightarrow{\text{model}} `` | ``wT[i + 1] `` |
| `Residual("KPP", KPP.Parameters())` | ``\text{KPP}(T[i])`` | ``\xrightarrow{\text{model}}`` | ``T[i] - \text{KPP}(T[i]) `` |  
| `Residual("TKE") TKEMassFlux.TKEParameters()`     | `` \text{TKE}(T[i]) `` | ``\xrightarrow{\text{model}}`` | ``T[i] - \text{TKE}(T[i]) `` |  

Where T[i] is a D-length vector of values from the horizontally-averaged temperature profile at time index t=i.
**Note that all model inputs are normalized using min-max scaling during pre-processing.** If the profile is a temperature profile,
this scaling is computed based on the profile at the first timestep.

See [OceanTurb.jl](https://github.com/glwagner/OceanTurb.jl) documentation for KPP and TKEMassFlux parameter options.

***

## Gaussian Process Regression

Gaussian Process (GP) regression produces a distribution over functions that interpolate the training data.
Here we concern ourselves with the mean GP prediction.

### Kernels

The kernel (or covariance) function sets the form of the interpolation function.

| Kernel ID | Name        | Parameters | Equation |
| :---:     |    :---     | :---       | :---     |
| 1         | Squared exponential     | γ: length-scale; σ: signal variance | ``k(x,x') = σ * exp( - ||x-x'||² / 2γ² ) `` |
| 2         | Matérn with ʋ=1/2       | γ: length-scale; σ: signal variance | ``k(x,x') = σ * exp( - ||x-x'|| / γ ) `` |
| 3         | Matérn with ʋ=3/2       | γ: length-scale; σ: signal variance | ``k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ) `` |
| 4         | Matérn with ʋ=5/2       | γ: length-scale; σ: signal variance | ``k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ) `` |
| 5         | Rational quadratic      | γ: length-scale; σ: signal variance ; α | ``k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α) `` |
| 1         | Squared exponential     | γ: length-scale; σ: signal variance | k(x,x') = σ * exp( - ||x-x'||² / 2γ² ) |
| 2         | Matérn with ʋ=1/2       | γ: length-scale; σ: signal variance | k(x,x') = σ * exp( - ||x-x'|| / γ ) |
| 3         | Matérn with ʋ=3/2       | γ: length-scale; σ: signal variance | k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ) |
| 4         | Matérn with ʋ=5/2       | γ: length-scale; σ: signal variance | k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ) |
| 5         | Rational quadratic      | γ: length-scale; σ: signal variance ; α | k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α) |

### Basic Example

In this example, we train our model and test our model on the data from the same simulation.
```julia
# problems
params   = KPP.Parameters( )
problem  = Residual("KPP", params)

# data
filename = "general_strat_32_profiles.jld2"
D        = 16
N        = 4

# kernel
k        = 1
logγ     = -3.0
logσ     = 0.0
distance = euclidean_distance
kernel   = get_kernel(k, logγ, logσ, distance)

# data
𝒟 = LearnConvection.Data.data(filename, problem; D=D, N=N)

# model
𝒢 = LearnConvection.GaussianProcess.model(𝒟; kernel = kernel)

# Animate the mean GP prediction.
anim = animate_profile(𝒢, 𝒟)
gif(anim, "animated_profile.gif")

```
