using Revise
using SparseArrayKit
using VectorInterface
using FoSpCore


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
V = U1FockSpace(geometry, 2, 6)
basis = all_states_U1(V)
lattice = Lattice(geometry)
t = ManyBodyTensor_init(ComplexF64, V, 1,1)
t2 =  ManyBodyTensor_init(ComplexF64, V, 2,2)
t.tensor .= randn_sparse(ComplexF64, Tuple(repeat([16], 2)), 0.1)
t2.tensor .= randn_sparse(ComplexF64, Tuple(repeat([16], 4)), 0.01)
sp = nbody_Op(V, lattice, t) 
tp =  nbody_Op(V, lattice, t2)
length(tp.terms)
O = sp + tp 
O = O + dagger_FO(O)
length(O.terms)
s = MutableFockVector(MutableFockState.(basis))
w = Base.copy(s)
@time for i in 1:10
    apply!(O, w, s);
end
@time begin 
    O_mf = transition_representation(O, basis);
length(O_mf.transitions)
sv = ones(ComplexF64, length(basis))
wv = similar(v)
end;
@time  for i in 1:10
    apply!(O_mf, wv, v)
end

w2 = [i.coefficient for i in w.vector]
norm(wv - w2)