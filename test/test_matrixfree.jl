using Test
using SparseArrayKit
using VectorInterface
using Revise
using FoSpCore

# --------------------------
# Utilities
# --------------------------
function randn_sparse(T::Type{<:Number}, sz::Dims, p=0.5)
    a = SparseArray{T}(undef, sz)
    for I in keys(a)
        if rand() < p
            a[I] = randn(T)
        end
    end
    return a
end

# --------------------------
# Setup
# --------------------------
geometry = (16,)
V = U1FockSpace(geometry, 2, 6)
basis = all_states_U1(V)
lattice = Lattice(geometry)

t = ManyBodyTensor_init(ComplexF64, V, 1, 1)
t2 = ManyBodyTensor_init(ComplexF64, V, 2, 2)

t.tensor .= randn_sparse(ComplexF64, Tuple(repeat([16], 2)), 0.1)
t2.tensor .= randn_sparse(ComplexF64, Tuple(repeat([16], 4)), 0.01)


# --------------------------
# Tests
# --------------------------
@testset "Many-body Operator Tests" begin
    sp = nbody_Op(V, lattice, t)
    tp = nbody_Op(V, lattice, t2)

    O = sp + tp
    # Test terms length
    @test length(tp.terms) > 0
    
    
    @test length(O.terms) == length(sp.terms) + length(tp.terms)
    global O
    # Test dagger / Hermitian combination
    O = O + dagger_FO(O)

    # Test applying to Fock vectors
    s = MutableFockVector(basis)
    w = Base.copy(s)
    for i in 1:2  # smaller loop for tests
        apply!(O, w, s)
    end
    @test length(w.vector) == length(s.vector)

    # Test matrix-free transition representation
    O_mf = transition_representation(O, basis)
    @test length(O_mf.transitions) == length(basis)
    
    # Test applying matrix-free operator to vector
    sv = ones(ComplexF64, length(basis))
    wv = similar(sv)
    apply!(O_mf, wv, sv)
    @test length(wv) == length(sv)

    # Compare with original vector
    w2 = [i.coefficient for i in w.vector]
    @test isapprox(norm(wv - w2), 0.0; atol=1e-10)

    # --------------------------
    # Optional timing (not part of tests)
    # --------------------------
    @info "Timing examples "
    @time for i in 1:10
        apply!(O, w, s)
    end

    @time begin
        O_mf = transition_representation(O, basis)
        sv = ones(ComplexF64, length(basis))
    end

    @time for i in 1:50
        wv = similar(sv)
        apply!(O_mf, wv, sv)
    end
end




@testset "MatrixFreeOperator tests" begin
    # Test getindex / setindex!
    A = zeroMFO(3)
    A[1,1] = 1 + 0im
    A[1,3] = 2 
    A[2,2] = 3. + 0im
    A[3,1] = 4.
    A[3,3] = 0 + 0im
    @test A[1,1] == 1 + 0im
    @test A[1,3] == 2 + 0im
    @test A[2,2] == 3 + 0im
    @test A[3,1] == 4 + 0im
    @test A[3,3] == 0 + 0im  # missing entry

    # Test overwrite and zero removal
    A[1,1] = 0 + 0im
    @test A[1,1] == 0 + 0im
    @test all(x[1] != 1 for x in A.transitions[1])  # should remove the zero entry

    # Test copy
    B = copy(A)
    A[1,3] = 0 + 0im
    @test B[1,3] == 2 + 0im  # deep copy

    # Test addition
    C = A + B
    @test C[2,2] == 6 + 0im  # 3 + 3
    @test C[3,1] == 8 + 0im  # 4 + 4

    # Test scalar multiplication
    D = 2 * B
    @test D[2,2] == 6 + 0im
    @test D[1,3] == 4 + 0im

    # Test matrix-vector multiplication
    x = [1+0im, 2+0im, 3+0im]
    y = B * x
    # B[1,3] = 2, B[2,2] = 3, B[3,1] = 4
    @test y[1] == 2*3  # 6
    @test y[2] == 3*2  # 6
    @test y[3] == 4*1  # 4

    # Test transpose
    T = transpose(B)
    @test T[3,1] == 2 + 0im
    @test T[2,2] == 3 + 0im
    @test T[1,3] == 4 + 0im

    # Test adjoint
    A_adj = adjoint(B)
    @test A_adj[3,1] == conj(2+0im)
    @test A_adj[2,2] == conj(3+0im)
    @test A_adj[1,3] == conj(4+0im)
end

