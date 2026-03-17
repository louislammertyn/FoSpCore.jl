####### functions of common operations in these calculations ########

begin
nbody_geometry(geometry::Tuple, n::Int) = (n==1) ? geometry : ( geometry |> collect |> g-> repeat(g,n) |> Tuple)
    
delta(i::Int,j::Int) = (i==j)

make_index(site_tuple::NTuple{N, NTuple{D, Int}}) where {D,N} = site_tuple |> collect .|> collect |> s -> vcat(s...) 

get_sites(latt::Lattice) = collect(keys(latt.sites))
get_sites_v(latt::Lattice) = collect(values(latt.sites))


# - Optional `canonicalize` (e.g., sort the tuple for symmetric n-body tensors)
# - Uses a fast tolerance check with abs2(v) ≤ tol^2 to skip tiny values
function fill_nbody_tensor(t_init::ManyBodyTensor,
                           lattice::Lattice,
                           fillingconditions::Tuple;
                           support=nothing)

    V       = t_init.V
    n       = t_init.domain + t_init.codomain
    tensor  = t_init.tensor
    T       = eltype(tensor)

    # Default support: full Cartesian product over sites
    # (collect keys once for consistent, stable iteration order)
    if support === nothing
        sites = collect(keys(lattice.sites))
        support = Base.Iterators.product(ntuple(_ -> sites, n)...)
    end

    # Iterate the chosen support
    for idx in support
        # Accumulate contributions from all fillingconditions
        acc = zero(T)
        @inbounds for f in fillingconditions
            v = f(idx_tuple)           
            acc += v
        end
        
        ind = make_index(idx_tuple)         
        # Sum into any existing value at this index
        tensor[ind...] += acc
        
    end

    return t_init
end

####### Different helpers that create common lists of indices involved in typical Operator terms #####


###### Onsite tensor index list ######
function Onsite_tensor_indices(latt::Lattice, n::Int)
    sites = get_sites(latt)  
    return ( ntuple(_ -> s, n) for s in sites )
end



###### Nearest-neighbor tensor index list ######
function NN_tensor_indices(latt::Lattice)
    sites = get_sites(latt)                           
    neighbours = latt.NN 
    return ((s, n) for s in sites, n in neighbours)
end

    


####### different conditions for periodic boundary conditions on NN hopping #######

periodic(i::Int, j::Int, L::Int) = delta(i, mod(j-1, L))

function neighbour(s1::NTuple{D, Int}, s2::NTuple{D, Int}, dim::Int) where {D}
    diff = collect(s2) .- collect(s1)
    cond = diff[dim]==1
    return delta(sum(diff),1) * cond
end

function periodic_neighbour(s1::NTuple{D, Int}, s2::NTuple{D, Int},
                            dim::Int, lattice::Lattice, geometry::Tuple) where {D}

    !(s1 ∈ lattice.NN[s2]) && return false

    # a periodic bond wraps: one site is at the boundary, the other at 1
    return (s1[dim] == geometry[dim] && s2[dim] == 1) ||
           (s2[dim] == geometry[dim] && s1[dim] == 1)
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