

include("./formula.jl")


function nor!(w, mchi)
    m, M = mchi
    return sqrt((M + Ecm!(w, M, m)) / (2M) + 0im)
end



function VLO!(w, mch, decons; n=2)
    Cij = [4 -sqrt(3 / 2) 0 sqrt(3 / 2)
        -sqrt(3 / 2) 3 3/sqrt(2) 0
        0 3/sqrt(2) 0 -3/sqrt(2)
        sqrt(3 / 2) 0 -3/sqrt(2) 3]
    f = decons

    v = zeros(ComplexF64, n, n)
    if typeof(f) == Vector{Float64}
        for i in 1:n
            for j in 1:n
                v[i, j] = Cij[i, j] * (1 / f[i] / f[j]) * (2w - mch[i][2] - mch[j][2]) * nor!(w, mch[i]) * nor!(w, mch[j])
            end
        end
    elseif typeof(f) == Matrix{Float64}
        for i in 1:n
            for j in 1:n
                v[i, j] = Cij[i, j] * (1 / f[i, j] / f[i, j]) * (2w - mch[i][2] - mch[j][2]) * nor!(w, mch[i]) * nor!(w, mch[j])
            end
        end
    end
    return -1 / 4 * v
end


"""Potential at next to leading order"""
function VNLO!(w, mπ, mK, mch, b::Dict, d::Dict, decons::Vector; n=2)
    # Dij
    b0, bD, bF = b[:b0], b[:bD], b[:bF]
    μ1 = sqrt(mK^2 + mπ^2)
    μ2 = sqrt(5mK^2 - 3mπ^2)
    μ3 = sqrt(4mK^2 - mπ^2)
    μ4 = sqrt(16mK^2 - 7mπ^2)
    D = [4*(b0+bD)*mπ^2 -sqrt(3 / 2)*(bD-bF)*μ1^2 -(4bD * mπ^2)/sqrt(3) sqrt(3 / 2)*(bD+bF)*μ1^2;
        -sqrt(3 / 2)*(bD-bF)*μ1^2 2*(2b0+3bD+bF)*mK^2 (bD+3bF)*μ2^2/(3sqrt(2)) 0;
        -(4bD * mπ^2)/sqrt(3) (bD+3bF)*μ2^2/(3sqrt(2)) (4/9)*(3b0*μ3^2+bD*μ4^2) -(bD - 3bF)*μ2^2/(3sqrt(2));
        sqrt(3 / 2)*(bD+bF)*μ1^2 0 -(bD - 3bF)*μ2^2/(3sqrt(2)) 2*(2b0+3bD-bF)*mK^2]

    # Lij
    d1, d2, d3, d4 = d[:d1], d[:d2], d[:d3], d[:d4]
    L = [-4d2+4d3+2d4 sqrt(3 / 2)*(d1+d2-2d3) -sqrt(3)d3 sqrt(3 / 2)*(d1-d2+2d3);
        sqrt(3 / 2)*(d1+d2-2d3) d1+3d2+2*(d3+d4) (d1-3d2+2d3)/sqrt(2) (6d2-3d3);
        -sqrt(3)*d3 (d1-3d2+2d3)/sqrt(2) 2*(d3+d4) (d1+3d2-2d3)/sqrt(2);
        sqrt(3 / 2)*(d1-d2+2d3) 6d2-3d3 (d1+3d2-2d3)/sqrt(2) -d1+3d2+2*(d3+d4)]

    v = zeros(ComplexF64, n, n)
    for i in 1:n
        for j in 1:n
            Ei = Ecm!(w, mch[i]...) # energy of initial meson
            Ej = Ecm!(w, mch[j]...) # energy of finial meson
            v[i, j] = (D[i, j] - 2L[i, j] * Ei * Ej) * nor!(w, mch[i]) * nor!(w, mch[j]) * (1 / (decons[i] * decons[j]))
        end
    end

    return v
end