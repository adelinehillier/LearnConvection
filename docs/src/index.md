```@meta
CurrentModule = LearnConvection
```

# LearnConvection.jl

```@index
```

```@autodocs
Modules = [LearnConvection]
```

<!-- ```@meta
CurrentModule = OceanConvect
```

```@docs
func(x)
``` -->

### Simulation data

`src/les` Large Eddy Simulations from [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)

* Run Oceananigans LES simulations using `src/les/run.jl`

* Harvest data from the output files in `src/les/data` using `get_les_data.jl`

### Physics-based parameterizations

`src/kpp` K-Profile Parameterizations from [OceanTurb.jl](https://github.com/glwagner/OceanTurb.jl)

`src/tke` Turbulent-Kinetic-Energy-based Parameterizations from [OceanTurb.jl](https://github.com/glwagner/OceanTurb.jl)

### Machine learning-based parameterizations

`src/gpr` Gaussian Process Regression.

- Applying GPR to predict the evolution of horizontally averaged temperature or temperature flux profiles from LES simulations.

- `sequential_T` <img src="https://render.githubusercontent.com/render/math?math=T[i] \xrightarrow{\text{GP}} T[i \+ 1]">

- `sequential_wT` <img src="https://render.githubusercontent.com/render/math?math=wT[i] \xrightarrow{\text{GP}} wT[i \+ 1]">

- `residual_T` <img src="https://render.githubusercontent.com/render/math?math=T[i] \xrightarrow{\text{GP}} (T[i+1]-T[i])/ \Delta{t'}"> where
  <img src="https://render.githubusercontent.com/render/math?math=\Delta{t'}=\Delta{t}/N^2">

- `residual_KPP` <img src="https://render.githubusercontent.com/render/math?math=\text{KPP}(T[i]) \xrightarrow{\text{GP}} \text{KPP}(T[i]) - T[i]">

- `residual_TKE` <img src="https://render.githubusercontent.com/render/math?math=\text{TKE}(T[i]) \xrightarrow{\text{GP}} \text{TKE}(T[i]) - T[i]">

... where T[i] is a D-length vector of values from the horizontally-averaged temperature profile at time index t=I, normalized during pre-processing (to range from 0 to 1) and scaled back up during post-processing.

`src/nn` Neural Networks.

* Applying neural networks to predict the evolution of horizontally averaged temperature profiles from LES simulations.


***

## Gaussian Process Regression
