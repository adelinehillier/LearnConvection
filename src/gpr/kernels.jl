"""
Constructors for covariance (kernel) functions.

      Constructor                   Description                                Isotropic/Anisotropic
    - SquaredExponentialI(γ,σ):     squared exponential covariance function    isotropic
    - ExponentialI(γ,σ):            exponential covariance function            isotropic
    - RationalQuadraticI():         rational quadratic covariance function     isotropic
    - Matern12I():                  Matérn covariance function with ʋ = 1/2.   isotropic
    - Matern32I():                  Matérn covariance function with ʋ = 3/2.   isotropic
    - Matern52I():                  Matérn covariance function with ʋ = 5/2.   isotropic

Distance metrics

    - euclidean_distance            l²-norm:  d(x,x') = || x - x' ||",
    - derivative_distance           H¹-norm:  d(x,x') = || diff(x)./diff(z) - diff(x')./diff(z) ||",
    - antiderivative_distance       H⁻¹-norm: d(x,x') = || diff(x).*diff(z) - diff(x').*diff(z) ||"
"""
abstract type Kernel end

#  *--*--*--*--*--*--*--*--*--*--*--*
#  | Isotropic covariance functions |
#  *--*--*--*--*--*--*--*--*--*--*--*

""" SquaredExponentialI(γ,σ): squared exponential covariance function, isotropic """
struct SquaredExponentialI{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
    "Distance metric"
    d::Function
end

# evaluates the kernel function for a given pair of inputs
function kernel_function(k::SquaredExponentialI; z=nothing)
    # k(x,x') = σ * exp( - d(x,x')² / 2γ² )
  evaluate(a,b) = k.σ * exp(- k.d(a,b,z)^2 / 2*(k.γ)^2 )
  return evaluate
end

struct RationalQuadraticI{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
    "Shape parameter"
    α::T
    "Distance metric"
    d::Function
end

function kernel_function(k::RationalQuadraticI; z=nothing)
    # k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)
    function evaluate(a,b)
        l = (k.γ)^2 # squared length scale
     return k.σ * (1+(a-b)'*(a-b)/(2*k.α*l))^(-k.α)
 end
  return evaluate
end

struct Matern12I{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
    "Distance metric"
    d::Function
end

function kernel_function(k::Matern12I; z=nothing)
    # k(x,x') = σ * exp( - ||x-x'|| / γ )
  evaluate(a,b) = k.σ * exp(- k.d(a,b,z) / k.γ )
  return evaluate
end

struct Matern32I{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
    "Distance metric"
    d::Function
end

function kernel_function(k::Matern32I; z=nothing)
    # k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)
    function evaluate(a,b)
        c = sqrt(3)*k.d(a,b,z)/k.γ
        return k.σ * (1+c) * exp(-c)
    end
  return evaluate
end

struct Matern52I{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
    "Distance metric"
    d::Function
end

function kernel_function(k::Matern52I; z=nothing)
    # k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)
    function evaluate(a,b)
        g = sqrt(5)*k.d(a,b,z)/k.γ
        h = 5*(k.d(a,b,z)^2)/(3*k.γ^2)
        return k.σ * (1+g+h) * exp(-g)
    end
  return evaluate
end


# % The hyperparameters are:
# %
# % hyp = [ log(w(:))
# %         log(m(:))
# %         log(sqrt(v(:))) ]

"""
https://github.com/alshedivat/gpml/blob/master/cov/covSM.m

# % For more details, see
# % [1] SM: Gaussian Process Kernels for Pattern Discovery and Extrapolation,
# %     ICML, 2013, by Andrew Gordon Wilson and Ryan Prescott Adams,
# % [2] SMP: GPatt: Fast Multidimensional Pattern Extrapolation with GPs,
# %     arXiv 1310.5288, 2013, by Andrew Gordon Wilson, Elad Gilboa,
# %     Arye Nehorai and John P. Cunningham, and
# % [3] Covariance kernels for fast automatic pattern discovery and extrapolation
# %     with Gaussian processes, Andrew Gordon Wilson, PhD Thesis, January 2014.
# %     http://www.cs.cmu.edu/~andrewgw/andrewgwthesis.pdf
# % [4] http://www.cs.cmu.edu/~andrewgw/pattern/.

"""
# function covSMfast(Q,hyp,x,z)
#     smp = Q<0
#     Q = abs(Q)
#     n,D = size(x); P = smp*D+(1-smp);               # dimensionality, P=D or P=1
#     w = exp(reshape(  hyp(         1:P*Q) ,P,Q));     # mixture weights
#     m = exp(reshape(  hyp(P*Q+    (1:D*Q)),D,Q));     # spectral means
#     v = exp(reshape(2*hyp(P*Q+D*Q+(1:D*Q)),D,Q));     # spectral variances
#
#     T = reshape(T,[],D);
#
#     if smp
#         h(t2v, tm) = exp(-0.5*t2v).*cos(tm);
#         K = 1; w = reshape(w,Q,P)'; m = reshape(m,Q,D)'; v = reshape(v,Q,D)';
#         for d=1:D
#             K = K .* ( h( (T(:,d).*T(:,d))*v(d,:), T(:,d)*m(d,:) )*w(d,:)' );
#     end
#     K = reshape(K.*ones(size(T,1),1),n,[]);
#     else
         # E = exp(-0.5*(T.*T)*v); H = E.*cos(T*m);
         # K = reshape(H*w',n,[]);
         # return K
#   end
# end
#
#
#
# function SM(Q,hyp,x,z)
#   Q = abs(Q)
#   n,D = size(x);                                  # dimensionality
#   w = exp(reshape(  hyp(         1:Q) ,1,Q));     # mixture weights
#   m = exp(reshape(  hyp(Q+    (1:D*Q)),D,Q));     # spectral means
#   v = exp(reshape(2*hyp(Q+D*Q+(1:D*Q)),D,Q));     # spectral variances
#
#   # w = exp(reshape(  hyp(         1) ,1,Q));     # mixture weights
#   # m = exp(reshape(  hyp(1+    (1:D)),D,Q));     # spectral means
#   # v = exp(reshape(2*hyp(1+16+ (1:D)),D,Q));     # spectral variances
#
#   T = reshape(T,[],D);
#   E = exp(-0.5*(T.*T)*v);
#   H = E.*cos(T*m);
#   K = reshape(H*w',n,[]);
#   return K
# end
#
#
# function SMP(Q,hyp,x,z)
#   Q = abs(Q)
#   n,D = size(x);                                    # dimensionality
#   w = exp(reshape(  hyp(         1:D*Q) ,D,Q));     # mixture weights
#   m = exp(reshape(  hyp(D*Q+    (1:D*Q)),D,Q));     # spectral means
#   v = exp(reshape(2*hyp(D*Q+D*Q+(1:D*Q)),D,Q));     # spectral variances
#
#   T = reshape(T,[],D);
#
#   K = 1;
#   w = reshape(w,Q,D)';
#   m = reshape(m,Q,D)';
#   v = reshape(v,Q,D)';
#
#   h(t2v, tm) = exp(-0.5*t2v).*cos(tm);
#   for d=1:D
#       K = K .* ( h( (T(:,d).*T(:,d))*v(d,:), T(:,d)*m(d,:) )*w(d,:)' );
#   end
#   K = reshape(K.*ones(size(T,1),1),n,[]);
#
#   return K
# end
