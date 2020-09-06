using JLD2
using Plots
using Printf

filename = "general_strat_8_profiles"
# filename = "general_strat_16_profiles"
# filename = "general_strat_32_profiles"

base_dir = pwd()*"/les_data_sandreza/"*filename
filename = base_dir*"/"*filename*".jld2"

les = jldopen(filename, "r"); 
les_keys = keys(les)
timeseries_keys = keys(les["timeseries"]["t"])

Nz = length(collect(les["grid"]["zC"]))
Nt = length(timeseries_keys)

## collect T and wT profile data
z = collect(les["grid"]["zC"])
T = zeros(Nz, Nt)
wT = zeros(Nz, Nt)

for j in 1:Nt
    key = timeseries_keys[j]
    @. T[:,j] = les["timeseries"]["T"][key][2:(end-1)]
    @. wT[:,j] = les["timeseries"]["wT"][key][2:(end-1)]
end

# daystring = @sprintf("%.3f", 123.0)

## plot temperature profile
xlims = (minimum(T), maximum(T))
anim = @animate for j = 1:Nt
    plot(T[:,j], z, legend = false, xlabel="T [°C]", ylabel="z [m]", xlims=xlims)
end
gif(anim, base_dir*"/z_vs_T.gif", fps = 50)
mp4(anim, base_dir*"/z_vs_T.mp4", fps = 50)

## plot temperature flux profile
xlims = (minimum(wT), maximum(wT))
anim = @animate for j = 1:Nt
    plot(wT[:,j], z, legend = false, xlabel="wT [m·°C/s²]", ylabel="z [m]", xlims=xlims)
end
gif(anim, base_dir*"/z_vs_wT.gif", fps = 20)
mp4(anim, base_dir*"/z_vs_wT.mp4", fps = 20)