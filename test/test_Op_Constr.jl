using Test
using Revise
using SparseArrayKit
using QuantumFockCore  


V = U1FockSpace((2,3), 5, 5)
lattice = Lattice((2,3))
# --------------------------------------------------
# TEST 1: ManyBodyTensor construction and indexing
# --------------------------------------------------
@testset "ManyBodyTensor basics" begin
    MBT = ManyBodyTensor(ComplexF64, V, 2, 1)  # 2 annihilation, 1 creation

    @test isa(MBT, AbstractArray)
    @test size(MBT) == (2,3,2,3,2,3)  # N=3 copies of D=2
    MBT.tensor[CartesianIndex(1,1,1,1,1,1)] = 1.0 + 0im
    @test MBT[1,1,1,1,1,1] == 1.0 + 0im
end

# --------------------------------------------------
# TEST 2: n_body_Op generates Fock operator from tensor
# --------------------------------------------------
@testset "n_body_Op" begin
    MBT = ManyBodyTensor(ComplexF64, V, 2, 1)  # 2 annihilation, 1 creation
    MBT.tensor[CartesianIndex(1,1,2,3, 1,1)] = 2.0 + 0im

    Op = n_body_Op(V, lattice, MBT)
    @test isa(Op, MultipleFockOperator) 
    # If operator is not zero, should contain one term with correct coefficient
    if isa(Op, MultipleFockOperator)
        @test length(Op.terms) == 1
        @test Op.terms[1].coefficient == 2.0 + 0im
        @test Op.terms[1].product == ((1,true),(6,false), (1, false))
    end
end
 
# --------------------------------------------------
# TEST 3: extract_n_body_tensors converts back
# --------------------------------------------------
@testset "extract_n_body_tensors" begin
    # Build a MultipleFockOperator manually
    term = FockOperator(((1,false),(2,true)), 3.0 + 0im, V)
    term2 = FockOperator(((1,false),(2,true), (4, false)), 3.0 + 0im, V)
    O = term + term2

    tensors = extract_n_body_tensors(O, lattice)
    @test length(tensors) == 2
    tensor = tensors[1]
    tensor2 = tensors[2]
    @test tensor.domain == 1
    @test tensor.codomain == 1
    @test tensor2.domain == 2
    @test tensor2.codomain == 1
    # Flattened indices corresponding to sites 1 and 2
    inds_flat = vcat(collect(lattice.sites_v[2]), collect(lattice.sites_v[1])) |> Tuple
    ci = CartesianIndex(inds_flat)
    @test tensor.tensor[ci] == 3.0 + 0im

    inds_flat2 = vcat(collect(lattice.sites_v[2]), collect(lattice.sites_v[1]), collect(lattice.sites_v[4])) |> Tuple
    ci2 = CartesianIndex(inds_flat2)
    @test tensor2.tensor[ci2] == 3.0 + 0im
end

# --------------------------------------------------
# TEST 4: Convert between vectorised and non vectorised tensors
# --------------------------------------------------
function randn_sparse(T::Type{<:Number}, sz::Dims, p=0.5)
    a = SparseArray{T}(undef, sz)
    for I in keys(a)
        if rand() < p
            a[I] = randn(T)
        end
    end
    return a
end

@testset "extract_n_body_tensors" begin
    V = U1FockSpace((2,3), 3, 3)
    lattice = Lattice((2,3))
    t = ManyBodyTensor(ComplexF64, V, 4, 3)
    t.tensor .= randn_sparse(ComplexF64, Tuple(repeat([2,3], 7)), 0.5)

    t_v = vectorize_tensor(t, lattice)
    @test t.tensor == devectorize_tensor(t_v, lattice)

    V = U1FockSpace((3,2), 3, 3)
    lattice = Lattice((3,2))
    tensor = randn_sparse(ComplexF64, Tuple(repeat([6], 7)), 0.5)

    t_v = ManyBodyTensor(tensor, V, 3, 4)
    t = devectorise_tensor(t_v, lattice)
    @test t_v.tensor == vectorize_tensor(t, lattice) 
end

V = U1FockSpace((2,3), 3, 3)
lattice = Lattice((2,3))
t = ManyBodyTensor(ComplexF64, V, 4, 3)
t.tensor .= randn_sparse(ComplexF64, Tuple(repeat([2,3], 7)), 0.5)
t
t_v = vectorize_tensor(t, lattice)
@test t.tensor == devectorize_tensor(t_v, lattice)