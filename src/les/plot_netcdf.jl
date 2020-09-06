using NetCDF, Plots, Printf

filename = "convection_day"

base_dir = pwd()*"/les_data/"*filename
filename = base_dir*"/"*filename*".nc"

filename = pwd() * sub_directory * "_middepth.nc"
ncinfo(filename)
x = Array(NetCDF.open(filename, "yC"))
y = Array(NetCDF.open(filename, "yC"))
z = Array(NetCDF.open(filename, "zC"))
t = Array(NetCDF.open(filename, "time"))
sim_day = t ./ 86400
z = (z[2:end] + z[1:end-1]) / 2
b = NetCDF.open(filename, "b")
u = NetCDF.open(filename, "u")
v = NetCDF.open(filename, "v")
w = NetCDF.open(filename, "w")
##

# T = b./(g*α)
# @. T /= 9.81 * 2e-4
T = b./(9.81 * 2e-4)

cmax = maximum(Array(b[2:end-1,2:end-1, 1, 1:10:end]))
cmin = minimum(Array(b[:,:, 1, 1:10:end]))
clims = (cmin, cmax)
anim = @animate for i in 1:1:length(b[1, 1, 1, :])
    b_array = Array(b[:, :, 1, i])
    # p1 = contourf(y, z, b_array, fill = true, linewidth = 0, color = :ocean, clim = (-0.0013871555898403098, -3.3776441214941526e-6))
    day_label = @sprintf("%.2f ", sim_day[i])
    p1 = contourf(x, y, b_array', 
    color = :ocean, title = "Buoyancy at Middepth at day " * day_label ,
     xlabel = "Meridional [m]", ylabel = "Depth [m]"
     , clims = clims)
end
gif(anim, filename *".gif", fps = 15)