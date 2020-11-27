## Convective adjustment

𝒟_train  = LearnConvection.Data.data(train, problem; D=16, N=N);

x = 𝒟_train.x[1300]

plott(x) = plot(x, collect(1:16), legend=false)
p(x) = plot!(x, collect(1:16), legend=false)

plott(x)


y = copy(x)
y[12]+=0.1
y[13]+=0.05
y[3]+=0.1
y[4]+=0.1
y[7]+=0.1

myplot = plott(y)
p(y)

png(myplot, "myyy.png")

r = copy(y)
convective_adjust!(r)

myplot2 = plott(r)
png(myplot2, "myyy3.png")

function convective_adjust!(x)
    # remove negative gradients from temperature profile
    for i in length(x)-2:-1:2
        if x[i] > x[i+1]
            if x[i-1] > x[i]; x[i] = x[i+1]
            else; x[i] = (x[i-1]+x[i+1])/2
            end
        end
    end
end
