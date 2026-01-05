using Test
using Plots
using SparseArrayKit 
using Revise
using FoSpCore  

# --------------------------
# Setup
# --------------------------
trms = []
ts = []
for L in 3:10
geometry = (L,);
V = U1FockSpace(geometry, 2, 6)
basis = all_states_U1(V)
lattice = Lattice(geometry)

t1_1 = ManyBodyTensor_rnd(ComplexF64, V, 1, 1, 0.1)
t2_1 = ManyBodyTensor_rnd(ComplexF64, V, 2, 2, 0.01)

t1_2 = ManyBodyTensor_rnd(ComplexF64, V, 1, 1, 0.1)
t2_2 = ManyBodyTensor_rnd(ComplexF64, V, 2, 2, 0.01)

O1 = nbody_Op(V, lattice, t1_1) + nbody_Op(V, lattice, t2_1)
O2 = nbody_Op(V, lattice, t1_2) + nbody_Op(V, lattice, t2_2)

N = (length(O1.terms) + length(O2.terms)) / 2
t1 = time()

@time o = commutator(O1,O2);
push!(ts,  time() - t1)
push!(trms, N)
end
ts
trms.^2
# --------------------------------------------------
# Setup Fock space and operator
# --------------------------------------------------
V = U1FockSpace((4,), 4, 4)

# Operator: 2 annihilation, 2 creation on site 1, 1 annihilation and 1 creation on site 2
O = FockOperator(((1, false), (1,false), (1, true), (1,true), (2,false), (2, true)), 2.0 + 0im, V)

# --------------------------------------------------
# Test: normal ordering
# --------------------------------------------------

f= n -> for i in 1:n; r = rand((0,1,2), 8); str = to_same_site_string(r); str2 = to_same_site_string(filter(x->x!=2, r)); @assert str.bits==str.bits;end
f(1000)
@testset "Normal ordering of FockOperator" begin
    result = normal_order(O)

    # Check type
    @test isa(result, MultipleFockOperator)

    # Check that cnumber is non-negative (scalar part from commutators)
    @test isa(result.cnumber, Number)

    # Check that each term is a FockOperator
    @test all(isa(t, FockOperator) for t in result.terms)

    # Check that coefficients are complex
    @test all(isa(t.coefficient, ComplexF64) for t in result.terms)
end

# --------------------------------------------------
# Test: adding scalar to MultipleFockOperator
# --------------------------------------------------
@testset "Adding MultipleFockOperator with scalar" begin
    mop = MultipleFockOperator([], 1.0)
    mop_plus_O = mop + O  # Should correctly add O to empty operator with scalar

    @test isa(mop_plus_O, MultipleFockOperator)
    @test mop_plus_O.cnumber == 1.0
    @test length(mop_plus_O.terms) == 1
    @test mop_plus_O.terms[1].product == O.product
    @test mop_plus_O.terms[1].coefficient == O.coefficient
end