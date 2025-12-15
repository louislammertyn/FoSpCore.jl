using Test, Revise
using FoSpCore

# Import everything from your module/code
 # Replace with actual file name if needed

@testset "FockSpace & FockState" begin
    u1_space = U1FockSpace((2,), 2, 2)
    unres_space = UnrestrictedFockSpace((2,), 3)

    fs1 = fock_state(u1_space, [1, 1])
    fs2 = fock_state(u1_space, [2, 0], 2.0 + 0im)
    fs3 = fock_state(u1_space, [2, 0], 1.0 + 0im)

    @test fs1.occupations == (1, 1)
    @test fs2.coefficient == 2.0 + 0im
    @test fs1 == fs1

    # Basic arithmetic
    s_add = fs2 + fs3
    @test s_add isa FockState
    @test s_add.coefficient == 3.0 + 0im

    s_diff = fs2 - fs3
    @test s_diff.coefficient == 1.0 + 0im

    # Multiplication (inner product)
    @test fs2 * fs3 == (2.0 + 0im)' * (1.0 + 0im)
    @test fs1 * fs2 == 0

    # Scaling
    @test (2 * fs1).coefficient == 2.0 + 0im

    # Check U(1) symmetry
    @test checkU1(fs1)

    # Norm
    @test norm2FS(fs2) == abs2(2.0)
end

@testset "MultipleFockState" begin
    u1_space = U1FockSpace(3, 2, 2)
    fs1 = fock_state(u1_space, [1, 1])
    fs2 = fock_state(u1_space, [2, 0])
    mfs = fs1 + fs2

    @test mfs isa MultipleFockState
    @test length(mfs.states) == 2

    mfs2 = fs2 + fs2  # Same occupation: should be a FockState
    @test mfs2 isa FockState
    @test mfs2.coefficient == 2.0 + 0im

    # Norm
    @test norm2FS(mfs) ≈ abs2(fs1.coefficient) + abs2(fs2.coefficient)
end

@testset "ZeroFockState" begin
    u1_space = U1FockSpace(3, 2, 2)
    fs = fock_state(u1_space, [1, 1])
    z = ZeroFockState()

    @test fs + z == fs
    @test fs - z == fs
    @test z + fs == fs
    @test z - fs == fs

    @test z * fs == 0
    @test fs * z == 0
    @test 3 * z == z
end

@testset "Creation and Annihilation Operators" begin
    space = UnrestrictedFockSpace(2, 3)
    fs = fock_state(space, [0, 2])

    fs_a1 = a_j(fs, 1)
    #@test fs_a1.coefficient == 0.0 + 0im
    @test fs_a1 isa ZeroFockState

    fs_a2 = a_j(fs, 2)
    @test fs_a2.coefficient ≈ sqrt(2)

    fs_ad1 = ad_j(fs, 1)
    @test fs_ad1.occupations == (1, 2)
    @test fs_ad1.coefficient ≈ sqrt(1)

    fs_ad2 = ad_j(fs, 2)
    @test fs_ad2.coefficient ≈ sqrt(3)
end

@testset "Fock Operators" begin
    op = FockOperator(((1, true), (2, false)), 1.0 + 0im)
    mop = MultipleFockOperator([op])

    # Operator + Operator
    op2 = FockOperator(((1, true), (2, false)), 2.0 + 0im)
    sumop = op + op2
    @test sumop isa FockOperator
    @test sumop.coefficient == 3.0 + 0im

    # Operator on FockState
    space = UnrestrictedFockSpace(2, 2)
    state = fock_state(space, [1, 1])
    result = op * state
    @test result isa FockState || result isa ZeroFockState || result isa MultipleFockState

    # Dagger operator
    op_dag = dagger_FO(op)
    @test op_dag.product == ((2, true), (1, false))
    @test op_dag.coefficient == (1.0 + 0im)'

    # Multiple operator acting
    mop = MultipleFockOperator([op, FockOperator(((1,false),), 0.5+0im)])
    result = mop * state
    @test result isa AbstractFockState
end

@testset "Lattice Mapping" begin
    L = (3, 3)
    idx = vectorise_lattice((1, 2), L)
    @test idx == 8

    map = lattice_vectorisation_map((2, 2))
    @test map[(0, 0)] == 1
    @test map[(1, 1)] == 4
end