include("../les/OceananigansData.jl")
include("../les/LESbraryData.jl")

prefix = pwd()*"/src/les/data/"

directories=Dict(
    "free_convection"          => "2daySuite/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
    "strong_wind"              => "2daySuite/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
    "strong_wind_no_coriolis"  => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128_averaged_statistics.jld2",
    "weak_wind_strong_cooling" => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
    "strong_wind_weak_cooling" => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
    "strong_wind_weak_heating" => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
)

# directories=Dict(
#
#     "free_convection" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     ),
#
#     "strong_wind" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     ),
#
#     "strong_wind_no_coriolis" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128_averaged_statistics.jld2",
#     ),
#
#     "weak_wind_strong_cooling" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     ),
#
#     "strong_wind_weak_cooling" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     ),
#
#     "strong_wind_weak_heating" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     ),
# )

"""
read_les_output(filename::String; days=0)
"""
function read_les_output(filename::String)

    if filename[end-2:end]==".nc"
        # println("Collecting NetCDF data.")
        # include("../les/convert_netcdf_to_data.jl")
        filename=prefix*"data_netcdf/$(filename[1:end-3])/$(filename)"
        return ReadNetCDF_OceananigansData(filename)
    elseif filename[1:13]=="general_strat"
        # println("Collecting JLD2 data.")
        # include("../les/convert_jld2_to_data.jl")
        filename=prefix*"general_strat_sandreza/$(filename[1:end-5])/$(filename)"
        return ReadJLD2_OceananigansData(filename)
    else
        filename=prefix*directories[filename]
        return ReadJLD2_LESbraryData(filename)
    end
end

# read_les_output("strong_wind_weak_heating"; days=2)
