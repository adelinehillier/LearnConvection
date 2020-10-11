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
