####### functions of common operations in these calculations ########

begin
nbody_geometry(geometry::Tuple, n::Int) = (n==1) ? geometry : ( geometry |> collect |> g-> repeat(g,n) |> Tuple)
    
delta(i::Int,j::Int) = (i==j)

make_index(site_tuple::NTuple{N, NTuple{D, Int}}) where {D,N} = site_tuple |> collect .|> collect |> s -> vcat(s...) 


#### Helper function to fill in ManyBodyTensor types for operator construction
function fill_nbody_tensor(t_init::ManyBodyTensor, lattice::Lattice, fillingconditions::Tuple )
    V = t_init.V
    n = t_init.domain + t_init.codomain
    
    tensor = t_init.tensor

    sites = keys(lattice.sites)

    for s_tuple in product(ntuple(_->sites, n)...)
        for f in fillingconditions
            value = f(s_tuple)
            isapprox(value, 0. +0im; atol=1e-5) && continue

            ind = make_index(s_tuple)

            tensor[ind...] = value 
        end
    end
    return t_init
end

####### different conditions for periodic boundary conditions on NN hopping #######

periodic(i::Int, j::Int, L::Int) = delta(i, mod(j-1, L))

function neighbour(s1::NTuple{D, Int}, s2::NTuple{D, Int}, dim::Int) where {D}
    diff = collect(s2) .- collect(s1)
    cond = diff[dim]==1
    return delta(sum(diff),1) * cond
end

function periodic_neighbour(s1::NTuple{D, Int}, s2::NTuple{D, Int},
                            dim::Int, lattice::Lattice) where {D}
    
    nns = lattice.NN
    !(s1 ∈ nns[s2]) && return false

    δ = mod(s2[dim] - s1[dim], lattice.geometry[dim])

    return δ == 1 
end


function helical_neighbour(s1::Tuple{Int, Int}, s2::Tuple{Int, Int}, dim::Int, L::Int)
    dim_ = mod(dim, 2) + 1
    return delta(s1[dim] , L ) * delta(s2[dim],1) * delta(s1[dim_],s2[dim_]-1)
end


function helical_periodic(s1::Tuple{Int,Int}, s2::Tuple{Int,Int}, geometry::Tuple{Int,Int})
    
    Lx = geometry[1]
    Ly = geometry[2]
    # Condition: s1 = (Lx,Ly) and s2 = (1,1)
    if s1 == (Lx, Ly) && s2 == (1, 1)
        return true
    
    else
        return false
    end
end


end