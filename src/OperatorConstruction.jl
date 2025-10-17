############################################################
# General N-body Fock operator from a tensor
############################################################
struct ManyBodyTensor{T, N} <: AbstractArray{T, N}
    tensor::SparseArray{T, N}
    V::AbstractFockSpace       # the fixed Fock space
    domain::Int                # number of copies in domain (annihilation)
    codomain::Int              # number of copies in codomain (creation)
end

# array interface
Base.size(M::ManyBodyTensor) = size(M.tensor)
Base.getindex(M::ManyBodyTensor, I...) = M.tensor[I...]
Base.IndexStyle(M::ManyBodyTensor) = IndexStyle(M.tensor)

function ManyBodyTensor(::Type{T}, V::AbstractFockSpace, domain::Int, codomain::Int) where {T}
    N = domain + codomain             # number of modes
    D = length(V.geometry)            # number of indices per mode
    tensor_rank = N * D               # total number of indices
    tensor_size = repeat(collect(V.geometry), N) |> Tuple
    tensor = SparseArray{T, tensor_rank}(undef, tensor_size)
    return ManyBodyTensor{T, tensor_rank}(tensor, V, domain, codomain)
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
        println(io, "  nonzero entries:")
        for (idx, val) in pairs(MBT.tensor)
            println(io, "    ", Tuple(idx), " => ", val)
        end
        println(io, "  total nonzeros: ", nnz_entries)
    end
end

"""
    n_body_Op(V::U1FockSpace, lattice::Lattice, tensor::AbstractArray{ComplexF64}, ops::Vector{Bool}) -> MultipleFockOperator

Constructs an N-body Fock operator from an N-body tensor.

# Arguments
- `V::U1FockSpace`: Fock space
- `lattice::Lattice`: lattice object mapping sites to local indices
- `tensor::AbstractArray{ComplexF64}`: N-body tensor
- `ops::Vector{Bool}`: length N, `true` for creation, `false` for annihilation

# Returns
- `MultipleFockOperator` corresponding to the tensor
"""
function n_body_Op(V::AbstractFockSpace, lattice::Lattice, tensor::ManyBodyTensor)
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
            println(op_tuple)
            Op += FockOperator(op_tuple, coeff, V)
        end
    end

    return typeof(Op)==MultipleFockOperator ? Op : MultipleFockOperator([Op], 0)
end

function extract_n_body_tensors(O::MultipleFockOperator, lattice::Lattice)
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
        tensor = ManyBodyTensor(ComplexF64, V, domain, codomain)
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
    return sum([n_body_Op(V, lattice, t) for t in tensors])
end

