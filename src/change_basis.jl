"""
This functionality implements operator transformations under either:

1. Projections onto a subset of the total single particle Hilbert space, or
2. Full unitary basis transformations on the many-body Fock operators.

The transformations are of the form:

    d†_α = Σ_i φ_i^α c†_i

where φ_i^α=⟨i|ϕ^α⟩ are either:

- The eigenstates |ϕ^α⟩ defining the subspace onto which one projects, or
- If they form an orthonormal set, the basis functions into which the Fock operators are transformed.

The matrix encoding the projection or transformation is denoted as:

    M_αi = φ_i^α

Please note the the i index labels the vectorised modes of the system and α labels the eigenstates |ϕ^α>
"""

function transform(O::MultipleFockOperator, lattice::Lattice, modes::Matrix{ComplexF64})
    if size(modes,1) == size(modes,2)
        @assert isapprox(modes * modes', I, atol=1e-12)
    end

    V = O.space
    new_geometry = (size(modes,1),)
    if typeof(V) == UnrestrictedFockSpace
        new_V = UnrestrictedFockSpace(new_geometry, V.cutoff)
    elseif typeof(V) == U1FockSpace
        new_V = U1FockSpace(new_geometry, V.cutoff, V.particle_number)
    end

    tnsrs = extract_n_body_tensors(O, lattice)
    new_O = ZeroFockOperator()

    for t_ in tnsrs
        t_v = vectorize_tensor(t_, lattice)

        dom = t_.domain
        codom = t_.codomain
        N = dom + codom

        # build index strings
        new_tensor_indices = [Char('a' + i - 1) for i in 1:N]
        old_tensor_indices = [Char('a' + N + i - 1) for i in 1:N]

        # start building the tensor contraction string
        tensor_str = "@tensor new_t[" * join(new_tensor_indices, ",") * "] := t_v[" * join(old_tensor_indices, ",") * "]"

        # multiply by modes for creation and annihilation indices
        for i in 1:dom
            tensor_str *= " * conj(modes[" * string(new_tensor_indices[i]) * "," * string(old_tensor_indices[i]) * "])"
        end
        for i in dom+1:N
            tensor_str *= " * modes[" * string(new_tensor_indices[i]) * "," * string(old_tensor_indices[i]) * "])"
        end

        # evaluate the tensor contraction
        @eval $tensor_str

        # devectorize back to lattice tensor
        new_t = devectorize_tensor(new_t, lattice)
        new_0 += n_body_Op(new_V, lattice, new_t)
    end

    return new_O
end

