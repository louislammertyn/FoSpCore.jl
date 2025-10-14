using Test
using SparseArrayKit  # if your code depends on it
using YourPackageName  # replace with the module containing FockOperator, MultipleFockOperator, normal_order, U1FockSpace

# --------------------------------------------------
# Setup Fock space and operator
# --------------------------------------------------
V = U1FockSpace((4,), 4, 4)

# Operator: 2 annihilation, 2 creation on site 1, 1 annihilation and 1 creation on site 2
O = FockOperator(((1, false), (1,false), (1, true), (1,true), (2,false), (2, true)), 2.0 + 0im, V)

# --------------------------------------------------
# Test: normal ordering
# --------------------------------------------------
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