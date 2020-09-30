using LearnConvection
using Plots

D=16
N=4

problem  = Residual("TKE", TKEMassFlux.TKEParameters())

println("hello")
