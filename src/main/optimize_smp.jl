using Optim

function optimize_SMP_kernel(𝒟_train, 𝒟_validate, 𝒟_test; Q=1)

    x0 = 0.1*ones(Q*3)

    function f(hyp)
        for h in hyp;
            if h < 0 # invalid parameter--signal variance can't be negative
                return Inf
            end
        end
        𝒢 = model(𝒟_train; kernel=SpectralMixtureProductI(hyp))
        get_me_true_check(𝒢, 𝒟_validate)
    end

    r = Optim.optimize(f, x0)
    params = Optim.minimizer(r)

    println(params)

    𝒢 = model(𝒟_train; kernel=SpectralMixtureProductI(params))

    println(get_me_true_check(𝒢, 𝒟_test))

    return SpectralMixtureProductI(params)
end

# function optimize_SMP_kernel(𝒟, x0 = 0.1*ones(length(𝒟.y[1]*3)))
#
#     D = length(𝒟.y[1])
#
#     function f(hyp)
#         ℳ = model(𝒟; kernel=SMP(hyp, D))
#         get_me_true_check(𝒢, 𝒟)
#     end
#
#     params = optimize(f, x0)
#
#     𝒢 = model(𝒟; kernel=SMP(params, D))
#     println("Error: $(get_me_true_check(𝒢, 𝒟))")
#
#     return SMP(params, D)
# end
