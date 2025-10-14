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
function n_body_Op(V::U1FockSpace, lattice::Lattice, tensor::ManyBodyTensor)
    N = tensor.domain + tensor.codomain
    
    tensor_geometry = size(tensor)
    D = length(tensor_geometry) ÷ N

    # Sanity checks
    @assert D * N == length(tensor_geometry) "Tensor rank mismatch"
    @assert all(tensor_geometry[1:D] .== V.geometry) "Tensor geometry mismatch"
    @assert all(all(tensor_geometry[1:D] .== tensor_geometry[(i-1)*D+1:i*D]) for i in 2:N) "Tensor is not N-body symmetric"

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

    return typeof(Op)==MultipleFockOperator ? Op : MultipleFockOperator([Op])
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

    return Nbody_tensors
end

############################################################
# Tensor extraction and momentum-space operators
############################################################
############################################################
# Extract 2-body tensor from a MultipleFockOperator
############################################################
"""
    get_tensor_2body(Op::MultipleFockOperator, lattice::Lattice) -> Array{ComplexF64,2D}

Constructs a 2-body tensor representation of terms of the form a†_i a_j
from a `MultipleFockOperator`. Ignores other types of terms.

Arguments:
- `Op`: MultipleFockOperator containing operator terms
- `lattice`: Lattice object mapping sites to indices

Returns:
- `tensor`: 2D (or 2*D-dimensional) array of complex coefficients
"""
function get_tensor_2body(Op::MultipleFockOperator, lattice::Lattice)
    map_v_s = lattice.sites_v
    V = Op.terms[1].space
    geometry = V.geometry
    two_body_geometry = Tuple(vcat(collect(geometry), collect(geometry)))
    tensor = zeros(ComplexF64, two_body_geometry...)

    # Loop over all terms
    for O in Op.terms
        # Select terms with exactly one creation and one annihilation operator
        if (length(O.product) == 2) & (O.product[1][2] & !O.product[2][2])
            s = map_v_s[O.product[1][1]]  # creation site
            n = map_v_s[O.product[2][1]]  # annihilation site
            ind = vcat(collect(s), collect(n))
            tensor[ind...] = O.coefficient
        end
    end

    return tensor
end

############################################################
# Extract 4-body tensor from a MultipleFockOperator
############################################################
"""
    get_tensor_4body(Op::MultipleFockOperator, lattice::Lattice) -> Array{ComplexF64,4D}

Constructs a 4-body tensor representation of terms of the form
a†_i a†_j a_k a_l from a `MultipleFockOperator`.

Arguments:
- `Op`: MultipleFockOperator containing operator terms
- `lattice`: Lattice object mapping sites to indices

Returns:
- `tensor`: 4D (or 4*D-dimensional) array of complex coefficients
"""
function get_tensor_4body(Op::MultipleFockOperator, lattice::Lattice)
    map_v_s = lattice.sites_v
    V = Op.terms[1].space
    geometry = V.geometry
    four_body_geometry = Tuple(vcat(collect(geometry), collect(geometry),
                                    collect(geometry), collect(geometry)))
    tensor = zeros(ComplexF64, four_body_geometry...)

    # Loop over all operator terms
    for O in Op.terms
        if length(O.product) == 4
            # Count creation and annihilation operators
            daggers = sum(p[2] for p in O.product[1:2])
            annih = sum(!p[2] for p in O.product[3:4])
            
            if (daggers == 2) & (annih == 2)
                # Map lattice sites to indices
                bra1 = map_v_s[O.product[1][1]]
                bra2 = map_v_s[O.product[2][1]]
                ket1 = map_v_s[O.product[3][1]]
                ket2 = map_v_s[O.product[4][1]]

                # Build tensor index and assign coefficient
                ind = vcat(collect(bra1), collect(bra2), collect(ket1), collect(ket2))
                tensor[ind...] = O.coefficient
            end
        end
    end

    return tensor
end
