############################################################
# General N-body Fock operator from a tensor
############################################################
struct ManyBodyTensor{T, N} <: AbstractArray{T, N}
    tensor::SparseArray{T, N}
    V::AbstractFockSpace       # the fixed Fock space
    domain::Int                # number of copies in domain (annihilation)
    codomain::Int              # number of copies in codomain (creation)

    function ManyBodyTensor(tensor::SparseArray{T, N}, V::AbstractFockSpace,
                            domain::Int, codomain::Int) where {T, N}

        new{T, N}(tensor, V, domain, codomain)
    end
end

# array interface
Base.size(M::ManyBodyTensor) = size(M.tensor)
Base.getindex(M::ManyBodyTensor, I...) = M.tensor[I...]
Base.IndexStyle(M::ManyBodyTensor) = IndexStyle(M.tensor)
==(M1::ManyBodyTensor, M2::ManyBodyTensor) = (M1.tensor == M2.tensor &&
                                                    M1.V == M2.V &&
                                                    M1.domain == M2.domain &&
                                                    M1.codomain == M2.codomain)


function ManyBodyTensor_init(::Type{T}, V::AbstractFockSpace, domain::Int, codomain::Int; v=false) where {T}
    N = domain + codomain             # number of modes
    D = length(V.geometry)            # number of indices per mode
    if v
        tensor_rank = N
        tensor_size = ntuple(_->prod(V.geometry), N)
    else
        tensor_rank = N * D               # total number of indices
        tensor_size = repeat(collect(V.geometry), N) |> Tuple
    end
    tensor = SparseArray{T, tensor_rank}(undef, tensor_size)
    if typeof(V)==U1FockSpace
        @assert domain==codomain "U1 symmetry is not respected with this tensor"
    end
    return ManyBodyTensor(tensor, V, domain, codomain)
end

function ManyBodyTensor_rnd(::Type{T}, V::AbstractFockSpace, domain::Int, codomain::Int, p=0.1; v=false) where {T}
    mbt = ManyBodyTensor_init(T, V, domain, codomain; v=v)
    mbt.tensor .= randn_sparse(T, size(mbt), p)
    return mbt
end

function randn_sparse(T::Type{<:Number}, sz::Dims, p=0.1)
    a = SparseArray{T}(undef, sz)
    for I in keys(a)
        if rand() < p
            a[I] = randn(T)
        end
    end
    return a
end



# similar for ManyBodyTensor
function Base.similar(M::ManyBodyTensor{T,N}, ::Type{S}=T) where {T,N,S}
    # create a sparse array of the same size as the underlying tensor
    new_tensor = spzeros(S, size(M.tensor))
    # return a new ManyBodyTensor with the same Fock space, domain, codomain
    return ManyBodyTensor{S,N}(new_tensor, M.V, M.domain, M.codomain)
end

function Base.show(io::IO, ::MIME"text/plain", MBT::ManyBodyTensor)
    println(io, "ManyBodyTensor of type ", typeof(MBT.tensor), 
            " with shape ", size(MBT.tensor))
    println(io, "  Fock space V: ", MBT.V)
    println(io, "  domain (annihilation copies): ", MBT.domain)
    println(io, "  codomain (creation copies): ", MBT.codomain)
    nnz_entries = nonzero_length(MBT.tensor)
    if nnz_entries == 0
        println(io, "  (all entries are zero)")
    else
        println(io, "  total nonzeros: ", nnz_entries, " or ", round(nnz_entries / length(MBT.tensor);digits=3), "%")
    end
end

"""
    nbody_Op(V::U1FockSpace, lattice::Lattice, tensor::AbstractArray{ComplexF64}, ops::Vector{Bool}) -> MultipleFockOperator

Constructs an N-body Fock operator from an N-body tensor.

# Arguments
- `V::U1FockSpace`: Fock space
- `lattice::Lattice`: lattice object mapping sites to local indices
- `tensor::AbstractArray{ComplexF64}`: N-body tensor
- `ops::Vector{Bool}`: length N, `true` for creation, `false` for annihilation

# Returns
- `MultipleFockOperator` corresponding to the tensor
"""
function nbody_Op(V::AbstractFockSpace, lattice::Lattice, tensor::ManyBodyTensor)
    N = tensor.domain + tensor.codomain
    
    tensor_geometry = size(tensor)
    D = length(tensor_geometry) ÷ N

    # Sanity checks
    @assert D * N == length(tensor_geometry) "Tensor rank mismatch"
    @assert all(tensor_geometry[1:D] .== V.geometry) "Tensor geometry mismatch"
    @assert all(all(tensor_geometry[1:D] .== tensor_geometry[(i-1)*D+1:i*D]) for i in 2:N) "Tensor is not N-body symmetric"
    typeof(V) == U1FockSpace && @assert tensor.domain == tensor.codomain "The tensor provided does not respect particle number conservation while U(1) symmetry is imposed!"
    map_v_s = lattice.sites_v
    sites = collect(keys(map_v_s))
    Op = ZeroFockOperator()
    
    iszero(tensor.tensor) && return Op

    # Generate all combinations of N sites
    site_combinations = Iterators.product(ntuple(_->sites, N)...)

    for combo in site_combinations
        # Flatten the indices for the N-body tensor
        inds_flat = reduce(vcat, (collect(map_v_s[s]) for s in combo)) |> Tuple
        ci = CartesianIndex(inds_flat)    # convert to CartesianIndex
        coeff = tensor.tensor[ci]         # access the sparse tensor
        if !iszero(coeff)
            # Pair each site with its corresponding creation/annihilation boolean
            op_tuple = [(combo[i], i<= tensor.codomain) for i in 1:N] |> Tuple
            Op += FockOperator(op_tuple, coeff, V)
        end
    end

    return typeof(Op)==MultipleFockOperator ? Op : MultipleFockOperator([Op], 0)
end

function extract_nbody_tensors(O::MultipleFockOperator, lattice::Lattice)
    V = O.terms[1].space
    map_v_s = lattice.sites_v
    O = normal_order(O)
    # Group operators by (domain, codomain) counts
    Operator_type_dict = Dict{Tuple{Int, Int}, Vector{AbstractFockOperator}}()
    for op in O.terms
        domain = 0
        codomain = 0
        for b in op.product
            b[2] ? (codomain += 1) : (domain += 1)
        end
        type_key = (domain, codomain)
        if type_key in keys(Operator_type_dict)
            push!(Operator_type_dict[type_key], op)
        else
            Operator_type_dict[type_key] = [op]
        end
    end

    # For each type construct a ManyBodyTensor
    Nbody_tensors = Vector{ManyBodyTensor}()
    for ((domain, codomain), ops_list) in Operator_type_dict
        tensor = ManyBodyTensor_init(ComplexF64, V, domain, codomain)
        for op in ops_list
            # Flatten all site indices and convert to CartesianIndex
            inds_flat = reduce(vcat, (collect(map_v_s[b[1]]) for b in op.product)) |> Tuple
            ci = CartesianIndex(inds_flat)
            tensor.tensor[ci] = op.coefficient           
        end
        push!(Nbody_tensors, tensor)
    end
    !iszero(O.cnumber) && push!(Nbody_tensors, O.cnumber)
    return Nbody_tensors
end

function construct_Multiple_Operator(V::AbstractFockSpace, lattice::Lattice, tensors::Vector{ManyBodyTensor})
    return sum([nbody_Op(V, lattice, t) for t in tensors])
end

function vectorize_tensor(M::ManyBodyTensor{T,N}, lattice::Lattice) where {T,N}
    mapping = lattice.sites
    # Get old tensor and size
    old_tensor = M.tensor
    old_size = size(old_tensor)
    D = length(M.V.geometry)
    
    (length(old_size) == (M.domain + M.codomain)) &&  return M

    # Determine new tensor size
    # assume mapping contains all possible multi-indices
    new_size = prod(M.V.geometry)
    new_rank = M.domain + M.codomain

    new_tensor = SparseArray{T, new_rank}(undef, ntuple(_->new_size, new_rank)) 

    # Iterate over all stored entries in sparse tensor
    for I in keys(old_tensor)
        site_tuples = split_tuple(Tuple(I), D)
        new_index = [mapping[s] for s in site_tuples]           # map multi-index to vectorized index
        new_tensor[new_index...] = old_tensor[I]                # match values
    end

    return ManyBodyTensor(new_tensor, M.V, M.domain, M.codomain)
end

function split_tuple(t::NTuple{N, T}, chunk::Int) where {N, T}
    @assert N % chunk == 0 "Tuple length must be divisible by chunk size"
    return ntuple(i -> t[(chunk*(i-1)+1):(chunk*i)], N ÷ chunk)
end

function devectorize_tensor(M::ManyBodyTensor{T,N}, lattice::Lattice) where {T,N}
    # Invert the mapping: α -> site tuple
    inv_mapping = lattice.sites_v

    # Old tensor and rank
    old_tensor = M.tensor
    D = length(M.V.geometry)  # dimension of each site

    # Size for the new tensor (lattice indices)
    tensor_rank = M.domain + M.codomain
    site_size = M.V.geometry
    new_size = repeat(collect(site_size), N) |> Tuple
    new_tensor = SparseArray{T, length(new_size)}(undef, new_size)

    # to ensure idempotent property
    (D * tensor_rank == length(size(M.tensor))) && return M

    # Iterate over stored entries in the vectorized tensor
    for I in keys(old_tensor)
        # I is a tuple of vectorized indices
        site_tuples = [collect(inv_mapping[i]) for i in Tuple(I)]          # map each α back to site tuple
        lattice_indices = reduce(vcat, site_tuples)        # flatten into a single tuple
        new_tensor[lattice_indices...] = old_tensor[I]     # copy value
    end

    return ManyBodyTensor(new_tensor, M.V, M.domain, M.codomain)
end









