# Normalizing the data after pre-processing and before post-processing.

abstract type Scaling end

# scales the data so that it ranges from 0 to 1.
struct min_max_scaling <: Scaling
    Δv::Float64 # max - min
    v_min::Float64
end

# scale: scales the data so that it ranges from 0 to 1.
scale(x, scaling::min_max_scaling) = (x .- scaling.v_min) ./ scaling.Δv

# unscale: undo scale
unscale(x, scaling::min_max_scaling) = (x .* scaling.Δv) .+ scaling.v_min

# struct min_max_scaling <: Scaling
#     ΔT::Float64 # range of Temp values in initial profile
#     T_max::Float64 # max value of the initial temperature profile (before cooling)
# end
#
# #scale: normalize the data so that it ranges from [-1,0]
# scale(x, scaling::min_max_scaling) = (x .- scaling.T_max) ./ scaling.ΔT
# #unscale: undo normalization
# unscale(x, scaling::min_max_scaling) = x .* scaling.ΔT .+ scaling.T_max

# struct wTscaling <: Scaling end
#
# #scale: divide all wT values by the maximum value across all timesteps and gridpoints
# scale(x, scaling::wTscaling) = (x .+ 1e-5) ./ 4e-5
# #scale: undo scale
# unscale(x, scaling::wTscaling) = x .* 4e-5 .- 1e-5

# struct wTscaling <: Scaling
#     nc::Float64 #normalization constant
# end
#
# #scale: divide all wT values by the maximum value across all timesteps and gridpoints
# scale(x, scaling::wTscaling) = x ./ scaling.nc
# #scale: undo scale
# unscale(x, scaling::wTscaling) = x .* scaling.nc

# function get_scaling(V_name, vavg)
#     if V_name=="T"; return Tscaling(vavg[1][end]-vavg[1][1], maximum(vavg[1])) # Tscaling(ΔT, T_max)
#     elseif V_name=="wT"; return wTscaling( maximum(maximum, vavg) ) # wTscaling(nc)
#     else; throw(error()) end
# end
