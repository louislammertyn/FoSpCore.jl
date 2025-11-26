using Revise
using SparseArrayKit
using VectorInterface
using QuantumFockCore


function randn_sparse(T::Type{<:Number}, sz::Dims, p=0.5)
    a = SparseArray{T}(undef, sz)
    for I in keys(a)
        if rand() < p
            a[I] = randn(T)
        end
    end
    return a
end

geometry = (16,)
V = U1FockSpace(geometry, 2, 5)
basis = all_states_U1(V)
lattice = Lattice(geometry)
t = ManyBodyTensor_init(ComplexF64, V, 1,1)
t2 =  ManyBodyTensor_init(ComplexF64, V, 2,2)
t.tensor .= randn_sparse(ComplexF64, Tuple(repeat([16], 2)), 0.1)
t2.tensor .= randn_sparse(ComplexF64, Tuple(repeat([16], 4)), 0.01)
O = nbody_Op(V, lattice, t) + nbody_Op(V, lattice, t2)

s = MutableFockVector(MutableFockState.(basis))
w = Base.copy(s)
buf1 = MutableFockState(basis[1])
typeof(buf1)
buf2 = MutableFockState(basis[1])
@time apply!(O, w, buf1, buf2, s);

N = length(basis)
A = rand(ComplexF64, N, N)
x = rand(ComplexF64, N)

@time A * x;
s2 = MultipleFockState(basis);


#@time w2 = O * s2 ;

remove_zeros(w2 - to_fock_state(w))