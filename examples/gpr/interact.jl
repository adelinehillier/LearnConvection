
using LearnConvection
using Interact, Blink, Plots

D = 16 # gridpoints
N = 4  # amount of training data

# hyperparameter slider ranges
γs = -3.0:0.1:3.0
σs = 0.0:0.1:2.0

# file to gather data from
filename = togglebuttons(OrderedDict("general_strat_16" =>"general_strat_16_profiles.jld2",
                                     "general_strat_32" =>"general_strat_32_profiles.jld2"),
                                     label="LES")

kpp_params   = KPP.Parameters( ) # KPP.Parameters( CSL = 1.0, CNL = 1.0, Cb_T = 1.0, CKE = 1.0 )
tke_params   = TKEMassFlux.TKEParameters( )

# which variable to explore / which problem to solve
problem = togglebuttons(OrderedDict("Sequential(T)" => Sequential("T"),
                            "Sequential(wT)" => Sequential("wT"),
                            "Sequential(dT)" => Sequential("dT"),
                             "Residual(KPP)" => Residual("KPP", kpp_params),
                             "Residual(TKE)" => Residual("TKE", tke_params)),
                            label="Problem")

# problem = togglebuttons(Dict("Sequential(T)" =>"Temperature [°C]",
#                             "wT"=>"Temperature flux [°C⋅m/s]"),
#                             label="profile")


γ1 = slider(γs, label="log length scale, log₁₀(γ)") # hyperparameter knob
σ1 = slider(σs, label="log signal variance, log₁₀(σ²)") # hyperparameter knob
time_slider = slider(1:40:(1000), label="time [s]") # time [s]

# distance metric
dist_metric = tabulator(OrderedDict("l²-norm"  =>  "l²-norm:  d(x,x') = || x - x' ||",
                                    "H¹-norm"  =>  "H¹-norm:  d(x,x') = || diff(x)./diff(z) - diff(x')./diff(z) ||",
                                    "H⁻¹-norm" =>  "H⁻¹-norm: d(x,x') = || diff(x).*diff(z) - diff(x').*diff(z) ||"
                                    ))
# kernel choice
kernel_id = tabulator(OrderedDict("Squared exponential"       => "Squared exponential kernel:           k(x,x') = σ * exp( - ||x-x'||² / 2γ² )",
                             "Matern 1/2"                => "Matérn with ʋ=1/2:                    k(x,x') = σ * exp( - ||x-x'|| / γ )",
                             "Matern 3/2"                => "Matérn with ʋ=3/2:                    k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)",
                             "Matern 5/2"                => "Matérn with ʋ=5/2:                    k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)",
                             "Rational quadratic w/ α=1" => "Rational quadratic kernel:            k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",
                             ))

get_data(filename::String, problem) = LearnConvection.Data.data(filename, problem; D=D, N=N)

get_gp(𝒟, k) = LearnConvection.GaussianProcess.model(𝒟; kernel = k)

function get_d(dist_metric)
    if dist_metric==1; return euclidean_distance end
    if dist_metric==2; return derivative_distance end
    if dist_metric==3; return antiderivative_distance
    else
        throw(error())
    end
end

function plot_kernel(𝒟::ProfileData, kernel::Kernel)
    kmat = [kernel_function(kernel; z=𝒟.zavg)(i,j) for i in 1:10:𝒟.Nt, j in 1:10:𝒟.Nt]# fill kernel mx with values
    return heatmap(kmat, title = "Covariance Matrix", xaxis=(:false), yaxis=(:flip, :false), clims=(0.0,100), legend=true)
end

# (model_output, gpr_prediction)
predictions(𝒢, 𝒟) = predict(𝒢, 𝒟; postprocessed="both")

function plot_profile_and_output(𝒢, 𝒟, time_index, gp_predictions)
    p1 = plot_profile(𝒢, 𝒟, time_index, gp_predictions[2])
    p2 = plot_model_output(𝒢, 𝒟, time_index, gp_predictions[1])
    return plot(p1, p2, layout=(@layout [a b]), size=(850,400))
end

#updating variables
#output                function                       args
𝒟                      = map(get_data,                filename, problem)
d                      = map(get_d,                   dist_metric)
k                      = map(get_kernel,              kernel_id, γ1, σ1, d)
k_plot                 = map(plot_kernel,             𝒟, k)
𝒢                      = map(get_gp,                  𝒟, k)
gp_predictions         = map(predictions,             𝒢, 𝒟)
profile_plot           = map(plot_profile_and_output, 𝒢, 𝒟, time_slider, gp_predictions)
log_error_plot         = map(plot_error_histogram,    𝒢, 𝒟, time_slider)
hyp_landscape          = map(plot_landscapes_compare_error_metrics, kernel_id, 𝒟, d, γs)

# layout
top    = vbox(hbox(filename, problem), hbox(kernel_id, dist_metric), hbox(k_plot, hyp_landscape))
middle = vbox(γ1, σ1, time_slider)
bottom = hbox(profile_plot, log_error_plot) # aligns horizontally
ui     = vbox(top, middle, bottom) # aligns vertically

# Blink GUI
window = Window()
body!(window, ui)
