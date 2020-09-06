using Printf
using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.OutputWriters
using Oceananigans.Diagnostics
using Oceananigans.Utils
using Oceananigans.AbstractOperations

arch = GPU()
FT   = Float64

# length of the simulation
days = 0.25
end_time = days*day

# output name
filename_1 = "free_convection_day$(days)"
# output directory
base_dir = "les_output/$(filename_1)/"

# NetCDF output writer can't make new directories or overwrite files by itself
try 
    mkdir(base_dir) # make output directory
catch SystemError 
    # directory exists, so remove .nc files within it
    cd(base_dir)
    run(`rm -f "*".nc`)
    cd("../../")
end

## grid
Lx = Ly = Lz = 100 # extent
Nx = Ny = Nz = 128 # make sure that gridpoints are a multiple of 16 for the GPU

topology = (Periodic, Periodic, Bounded) # horizontally periodic BCs
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))
Hz = grid.Hz # number of halo points

## rotation
f = 1e-4 # coriolis parameter (positive - northern hemisphere)
coriolis = FPlane(FT, f=f)

# α = 2e-4  # Thermal expansion coefficient [K⁻¹]
# eos = LinearEquationOfState(FT, α=α, β=0)
buoyancy = BuoyancyTracer()

κh = νh = 1e-5   # Horizontal diffusivity and viscosity [m²/s]
κv = νv = 1e-5   # Vertical diffusivity and viscosity [m²/s]

N_s = 1e-2 * 2e-3  # Uniform vertical stratification [s⁻¹]
buoyancy_flux = 1.96e-7 / 4 # bouyancy flux

## closure
# closure = ConstantAnisotropicDiffusivity(FT, νh=νh, νv=νv, κh=κh, κv=κv)
closure = AnisotropicMinimumDissipation(FT)

#  *--*--*--*--*--*--*--*--*
#  |  Boundary Conditions  |
#  *--*--*--*--*--*--*--*--*

top_b_bc = FluxBoundaryCondition(buoyancy_flux)
bottom_b_bc = GradientBoundaryCondition(N_s)
b_bcs = TracerBoundaryConditions(grid, top = top_b_bc, bottom = bottom_b_bc)
top_C_bc = ValueBoundaryCondition(1.0)
C_bcs = TracerBoundaryConditions(grid, top=top_C_bc)
bcs = (b = b_bcs,)

#  *--*--*--*
#  | Model  |
#  *--*--*--*

model = IncompressibleModel(
            architecture = arch,
                float_type = FT,
                    grid = grid,
                coriolis = coriolis,
                buoyancy = buoyancy,
                closure = closure,
                tracers = (:b,),
    boundary_conditions = bcs
)

#  *--*--*--*--*--*--*--*--*
#  |  Initial Conditions   |
#  *--*--*--*--*--*--*--*--*

T_s  = 12.0    # Surface temperature [°C]
ε(σ) = σ * randn()
B₀(x, y, z) = N_s * (z + 100) + ε(1e-8) + 20*2e-3
set!(model, b=B₀) # set ICs

#  *--*--*--*--*
#  | Timestep  |
#  *--*--*--*--*

# Δt= 1.0
Δt_wizard = TimeStepWizard(cfl=0.3, Δt = 1.0, max_change=1.0, max_Δt = 3.0)
cfl = AdvectiveCFL(Δt_wizard)

#  *--*--*--*--*--*
#  | Simulation   |
#  *--*--*--*--*--*

# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 50

function print_progress(simulation)
    model = simulation.model
    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)

    @printf("[%05.2f%%] i: %d, t: %.2e days, umax: (%6.3e, %6.3e, %6.3e) m/s, CFL: %6.4e, next Δt: %.1e s\n",
    	    progress, i, t / day, umax, vmax, wmax, cfl(model), Δt_wizard.Δt)
end

simulation = Simulation(model, Δt=Δt_wizard, stop_time=end_time, progress=print_progress, progress_frequency=Ni)

u = model.velocities.u
v = model.velocities.v
w = model.velocities.w
b = model.tracers.b

#  *--*--*--*--*--*--*
#  | Output Writers  |
#  *--*--*--*--*--*--*
#  Set up output writers to write output variables to NetCDF (.nc) files

global_attributes = Dict(
    "f"                 => Dict("longname" => "Coriolis parameter", "units" => "---"),
    "model.closure.ν"   => Dict("longname" => "Coriolis parameter", "units" => "---"),
    "model.closure.ν"   => Dict("longname" => "Viscosity", "units" => "---"),
    "model.closure.κ.b" => Dict("longname" => "diffusivity", "units" => "---"),
    "Q_str"             => Dict("longname" => "Net surface heat flux into the ocean (+=down), >0 increases theta", "units" => "W/m²"),
    "f"                 => Dict("longname" => "Coriolis parameter", "units" => "---"),
    "rho"               => Dict("longname" => "Density", "units" => "---"),
    "ρ₀"                => Dict("longname" => "Density", "units" => "---")
    # file["parameters/viscosity"] = model.closure.ν
    # file["parameters/diffusivity"] = model.closure.κ.b
    # file["parameters/diffusivity_T"] = model.closure.κ.b
    # file["parameters/diffusivity_S"] = model.closure.κ.S
    # file["parameters/surface_cooling"] = Q_str
    # file["parameters/temperature_gradient"] = ∂θ∂z
    # file["boundary_conditions/top/FT"] = Fθ_str
    # file["boundary_conditions/top/Fu"] = Fu_str
    # file["boundary_conditions/bottom/dTdz"] = ∂θ∂z
)

#every output (each key in profiles) must have an associated attribute
output_attributes = Dict(
     "u"  => Dict("longname" => "Velocity in the x-direction", "units" => "m/s"),
     "v"  => Dict("longname" => "Velocity in the y-direction", "units" => "m/s"),
     "w"  => Dict("longname" => "Velocity in the z-direction", "units" => "m/s"),
     "b" => Dict("longname" => "Buoyancy",                    "units" => "m/s²"),
     "nu" => Dict("longname" => "Eddy viscosity",              "units" => "m²/s"), #####
 "kappaT" => Dict("longname" => "Eddy diffusivity of conservative temperature", "units" => "m²/s"),
     "uu" => Dict("longname" => "Velocity covariance between u and u", "units" => "m²/s²"),
     "vv" => Dict("longname" => "Velocity covariance between v and v", "units" => "m²/s²"),
     "ww" => Dict("longname" => "Velocity covariance between w and w", "units" => "m²/s²"),
     "uv" => Dict("longname" => "Velocity covariance between u and v", "units" => "m²/s²"),
     "uw" => Dict("longname" => "Velocity covariance between u and w", "units" => "m²/s²"),
     "vw" => Dict("longname" => "Velocity covariance between v and w", "units" => "m²/s²"),
     "wT" => Dict("longname" => "Vertical turbulent heat flux", "units" => "K*m/s"),
 "vshear" => Dict("longname" => "Conservative temperature",    "units" => "°C")
)

#  *--*--*--*--*--*--*--*--*--*--*--*
#  |  NetCDF Output Writer: Fields  |
#  *--*--*--*--*--*--*--*--*--*--*--*

fields = Dict(
    "u" => u,
    "v" => v,
    "w" => w,
    "b" => b
)

# field_writer = NetCDFOutputWriter(model, fields; 
#                                   filename=base_dir*filename_1*"_fields.nc",
#                                 # global_attributes=global_attributes,
#                                 # output_attributes=output_attributes, # use default output_attributes
#                                 # dimensions = field_dims, # computes field variable dimensions automatically
#                                   interval=2hour)

# simulation.output_writers[:field_writer] = field_writer

#  *--*--*--*--*--*--*--*--*--*--*--*
#  | NetCDF Output Writer: Profiles |
#  *--*--*--*--*--*--*--*--*--*--*--*

# diagnostics 
 Up    = HorizontalAverage(u;                        return_type=Array)
 Vp    = HorizontalAverage(v;                        return_type=Array)
 Wp    = HorizontalAverage(w;                        return_type=Array)
 Tp    = HorizontalAverage(b;                        return_type=Array)
 νp    = HorizontalAverage(model.diffusivities.νₑ;   return_type=Array)
κTp    = HorizontalAverage(model.diffusivities.κₑ.b; return_type=Array)
uu     = HorizontalAverage(u*u, model;               return_type=Array)
vv     = HorizontalAverage(v*v, model;               return_type=Array)
ww     = HorizontalAverage(w*w, model;               return_type=Array)
uv     = HorizontalAverage(u*v, model;               return_type=Array)
uw     = HorizontalAverage(u*w, model;               return_type=Array)
vw     = HorizontalAverage(v*w, model;               return_type=Array)
wT     = HorizontalAverage(w*b, model;               return_type=Array)
vshear = HorizontalAverage(∂z(u)^2 + ∂z(v)^2, model; return_type=Array)

# output variables
profiles = Dict(
     "u"  => model ->  Up(model)[1,1,1+Hz:end-Hz],
     "v"  => model ->  Vp(model)[1,1,1+Hz:end-Hz],
     "w"  => model ->  Wp(model)[1,1,1+Hz:end-Hz],
     "b"  => model ->  Tp(model)[1,1,1+Hz:end-Hz],
     "nu" => model ->  νp(model)[1,1,1+Hz:end-Hz],
 "kappaT" => model -> κTp(model)[1,1,1+Hz:end-Hz],
     "uu" => model ->  uu(model)[1,1,1+Hz:end-Hz],
     "vv" => model ->  vv(model)[1,1,1+Hz:end-Hz],
     "ww" => model ->  ww(model)[1,1,1+Hz:end-Hz],
     "uv" => model ->  uv(model)[1,1,1+Hz:end-Hz],
     "uw" => model ->  uw(model)[1,1,1+Hz:end-Hz],
     "vw" => model ->  vw(model)[1,1,1+Hz:end-Hz],
     "wT" => model ->  wT(model)[1,1,1+Hz:end-Hz],
 "vshear" => model -> vshear(model)[1,1,1+Hz:end-Hz]
)

profile_dims = Dict(k => ("zC",) for k in keys(profiles))
profile_dims["w"] = ("zF",)


profile_writer = NetCDFOutputWriter(model, profiles; 
                                    filename = base_dir*filename_1*"_profiles.nc",
                                    # global_attributes = global_attributes,
                                    output_attributes = output_attributes,
                                    dimensions = profile_dims,
                                    interval = 1hour)

simulation.output_writers[:profile_writer] = profile_writer

#  *--*--*--*--*--*--*
#  | Run simulation  |
#  *--*--*--*--*--*--*

# time_step!(model, 1)
run!(simulation)