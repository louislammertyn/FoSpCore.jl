using Revise
using FoSpCore


O1 = FockOperator(((1, false), (1, true), (2,true), (2, true)), 2. + 1im)
O2 = FockOperator(((2, true), (2, false), (3,true), (3, true)), 2. + 1im)
mo = O1 + O2
normal_order(mo)
commutator(O1, O2)

ad = FockOperator(((1, true),), 2. + 0im)
a = FockOperator(((1, false),), 1. + 0im)

V = U1FockSpace(3, 3, 6)
s = fock_state(V, [1, 2, 3], 2.)
commutator(a, ad) * s
