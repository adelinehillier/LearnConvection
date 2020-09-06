
"""
From sandreza/Learning/sandbox/oceananigans_converter.jl
  https://github.com/sandreza/Learning/blob/master/sandbox/oceananigans_converter.jl

custom_avg(Φ, n)
# Description
- Average a field down to n.
- Requires field to have evenly spaced points. Size of N leq length(Φ).
- Furthermore requires
# Arguments
- `Φ` :(vector) The field, an array
- `n` :(Int) number of grid points to average down to.
# Return
- `Φ2` :(vector) The field with values averaged, an array
"""
function custom_avg(Φ, n)
  m = length(Φ)
  scale = Int(floor(m/n))
  if ( abs(Int(floor(m/n)) - m/n) > eps(1.0))
      return error
  end
  Φ2 = zeros(n)
  for i in 1:n
      Φ2[i] = 0
          for j in 1:scale
              Φ2[i] += Φ[scale*(i-1) + j] / scale
          end
  end
  return Φ2
end