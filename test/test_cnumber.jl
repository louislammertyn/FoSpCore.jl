using Test
using Revise
using FoSpCore
# --------------------------------------------------
# TEST 1: Multiplication of FockOperators
# --------------------------------------------------
@testset "FockOperator multiplication" begin
    V = U1FockSpace((2,2), 3,3)
    op1 = FockOperator(((1,true),), 2.0+0im, V)
    op2 = FockOperator(((2,false),), 3.0+0im, V)

    prod = op1 * op2
    @test prod.coefficient == 6.0+0im
    @test prod.product == ((1,true),(2,false))
end

# --------------------------------------------------
# TEST 2: Multiplication of MultipleFockOperators
# --------------------------------------------------
@testset "MultipleFockOperator multiplication" begin
    V = U1FockSpace((2,2), 3,3)
    op1 = FockOperator(((1,true),), 2.0+0im, V)
    op2 = FockOperator(((2,false),), 3.0+0im, V)

    mop1 = MultipleFockOperator([op1], 1.0)
    mop2 = MultipleFockOperator([op2], 2.0)

    mop_prod = mop1 * mop2
    @test mop_prod.cnumber == 2.0
    # Check that terms include O1*O2, O1*cnumber, O2*cnumber
    coeffs = [t.coefficient for t in mop_prod.terms]
    println(mop_prod)
    @test all(x -> x in [6.0+0im, 4.0+0im, 3.0+0im], coeffs)
end

# --------------------------------------------------
# TEST 3: Multiplication with scalar
# --------------------------------------------------
@testset "Scalar multiplication with cnumber" begin
    mop = MultipleFockOperator([op1], 3.0)
    mop2 = 2 * mop
    @test mop2.cnumber == 6.0
    @test mop2.terms[1].coefficient == 4.0+0im
end