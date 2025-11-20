# We add a seperate Mutable FockState for performance critical codes 
mutable struct MutableFockState <: AbstractFockState
    occupations::Vector{Int}
    coefficient::ComplexF64
    space::AbstractFockSpace
    iszero::Bool
end

mutable struct MultipleMutableFockState <: AbstractFockState
    states::Vector{MutableFockState}
end



==(s1::MutableFockState, s2::MutableFockState) =
    s1.occupations == s2.occupations &&
    s1.coefficient == s2.coefficient &&
    s1.space == s2.space &&
    s1.iszero == s2.iszero


function Base.show(io::IO, s::MutableFockState)
    occ_str = join(s.occupations, ", ")
    coeff_str = s.coefficient == 1 + 0im ? "" : string("($(s.coefficient))", " ⋅ ")
    print(io, coeff_str * "| ", occ_str, " ⟩")
end

# MutableFockstate operations

Base.:*(c::Number, mfs::MutableFockState) = MutableFockState(copy(mfs.occupations), c * mfs.coefficient, mfs.space, mfs.iszero)
Base.:*(mfs::MutableFockState, c::Number) = c * mfs
Base.:*(f::FockState, mfs::MutableFockState) = f.coefficient' * mfs.coefficient
Base.:*(mfs::MutableFockState, f::FockState) = f.coefficient * mfs.coefficient'

function mul_Mutable!(c::Number, mfs::MutableFockState) 
    mfs.coefficient *= c
    mfs.iszero = zero(mfs.coefficient)
end
function mu_Mutable!(mfs::MutableFockState, c::Number) 
    mul_Mutable!(c, mfs)
end


###################  MutableFockState functionalities ###################
function MutableFockState(fs::FockState)
    return MutableFockState(collect(fs.occupations), fs.coefficient, fs.space, iszero(fs.coefficient))
end

function to_fock_state(mfs::MutableFockState)
    return FockState(ntuple(i -> mfs.occupations[i], length(mfs.occupations)), mfs.coefficient, mfs.space)
end

function reset2!(state::MutableFockState, occs::NTuple{N, Int}, coeff::ComplexF64) where N
    @inbounds for i in eachindex(occs)
        state.occupations[i] = occs[i]
    end
    state.iszero= iszero(coeff)
    state.coefficient = coeff
    return nothing
end
function reset!(state::MutableFockState, occs::Vector{Int}, coeff::ComplexF64) 
    state.occupations = occs
    state.coefficient = coeff
end

function norm2FS(mfs::MutableFockState)
    return abs2(mfs.coefficient)
end

cleanup_FS(mfs::MutableFockState) = mfs.coefficient == 0 ? ZeroFockState() : to_fock_state(mfs)


function a_j!(state::MutableFockState, j::Int)
    n = state.occupations[j]
    if n == 0
        state.coefficient = 0.0
        state.iszero = true
    else
        state.coefficient *= sqrt(n)
        state.occupations[j] -= 1
    end
end

function ad_j!(state::MutableFockState, j::Int)
    n = state.occupations[j]
    if n + 1 > state.space.cutoff
        state.coefficient = 0.0
        state.iszero = true
    else
        state.coefficient *= sqrt(n + 1)
        state.occupations[j] += 1
        
    end
end


###################### VectorInterface.jl compatibility for MultipleFockState ################

function LinearAlgebra.dot(x::MultipleMutableFockState, y::MultipleMutableFockState) 
    sum = 0
    @inbounds for i in eachindex(x.states)
        (x.states[i].iszero && y.states[i].iszero) && continue 
        sum += conj(x.states[i].coefficient) * y.states[i].coefficient 
    end
    return sum
end

LinearAlgebra.norm(x::MultipleFockState) = norm(x.coefficient)



##############################
# Scalar type
##############################

VectorInterface.scalartype(::MutableFockState) = ComplexF64
VectorInterface.scalartype(::MultipleMutableFockState) = ComplexF64

##############################
# Zero vectors
##############################


# In-place zero for MultipleMutableFockState
function VectorInterface.zerovector!(v::MultipleMutableFockState)
    empty!(v.states)
    return v
end

# BangBang for container: just call in-place if mutable
VectorInterface.zerovector!!(v::MultipleMutableFockState) = VectorInterface.zerovector!(v)

# Out-of-place zero
function VectorInterface.zerovector(v::MultipleMutableFockState)
    return MultipleMutableFockState(Vector{MutableFockState}())
end

##############################
# Scaling
##############################

function VectorInterface.scale!(s::MutableFockState, α)
    s.coefficient *= α
    s.iszero = s.coefficient == 0
    return s
end

VectorInterface.scale!!(s::MutableFockState, α) = VectorInterface.scale!(s, α)
VectorInterface.scale(s::MutableFockState, α) = MutableFockState(s.occupations, s.coefficient * α, s.space, s.iszero)

function VectorInterface.scale!(v::MultipleMutableFockState, α)
    for s in v.states
        VectorInterface.scale!(s, α)
    end
    return v
end

VectorInterface.scale!!(v::MultipleMutableFockState, α) = VectorInterface.scale!(v, α)
VectorInterface.scale(v::MultipleMutableFockState, α) = MultipleMutableFockState([scale(s, α) for s in v.states])
function VectorInterface.scale!(w::MultipleMutableFockState, v::MultipleMutableFockState, α) 
    w.states = scale(v,α)
    return w
end

    

##############################
# Addition
##############################

function VectorInterface.add!(w::MultipleMutableFockState, v::MultipleMutableFockState; α=1, β=1)
    @assert length(w.states) == length(v.states)
    for i in eachindex(w.states)
        w_i, v_i = w.states[i], v.states[i]
        w_i.coefficient = β * w_i.coefficient + α * v_i.coefficient
        w_i.iszero = w_i.coefficient == 0
    end
    return w
end

VectorInterface.add!!(w::MultipleMutableFockState, v::MultipleMutableFockState; α=1, β=1) =
    VectorInterface.add!(VectorInterface.zerovector(w), v; α=α, β=β)

VectorInterface.add(w::MultipleMutableFockState, v::MultipleMutableFockState; α=1, β=1) =
    VectorInterface.add!(VectorInterface.zerovector(w), v; α=α, β=β)

##############################
# Inner product
##############################

function VectorInterface.inner(v::MultipleMutableFockState, w::MultipleMutableFockState)
    @assert length(v.states) == length(w.states)
    s = zero(ComplexF64)
    for i in eachindex(v.states)
        s += v.states[i].coefficient * conj(w.states[i].coefficient)
    end
    return s
end

##############################
# Norm
##############################

function VectorInterface.norm(v::MultipleMutableFockState)
    sqrt(sum(abs2(s.coefficient) for s in v.states))
end
