# We add a seperate Mutable FockState for performance critical codes 
mutable struct MutableFockState <: AbstractFockState
    occupations::Vector{UInt8}
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

# Single mutable Fock state
function Base.copy(f::MutableFockState)
    MutableFockState(
        copy(f.occupations),   # copy the vector so it's independent
        f.coefficient,         # primitive types are copied automatically
        f.space,               # assume space is immutable/shared
        f.iszero               # Bool is primitive
    )
end

# Multiple mutable Fock states
function Base.copy(mf::MultipleMutableFockState)
    MultipleMutableFockState(
        [copy(s) for s in mf.states]   # copy each MutableFockState
    )
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
    return MutableFockState(UInt8.(collect(fs.occupations)), fs.coefficient, fs.space, iszero(fs.coefficient))
end

function to_fock_state(mfs::MutableFockState)
    return FockState(ntuple(i -> mfs.occupations[i], length(mfs.occupations)), mfs.coefficient, mfs.space)
end

function reset2!(state::MutableFockState, occs::NTuple{N, Int}, coeff::ComplexF64) where N
    @inbounds for i in eachindex(occs)
        state.occupations[i] = UInt8(occs[i])
    end
    state.iszero= iszero(coeff)
    state.coefficient = coeff
    return nothing
end
function reset!(state::MutableFockState, occs::Vector{UInt8}, coeff::ComplexF64) 
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

function a_j!(state1::MutableFockState, state2::MutableFockState, j::Int)
    n = state2.occupations[j]

    if n == 0
        # Resulting state is zero
        state1.coefficient = 0.0
        state1.iszero = true
        return state1
    end

    # Copy occupations from state2 → state1
    @inbounds for k in eachindex(state2.occupations)
        state1.occupations[k] = state2.occupations[k]
    end

    # Apply a_j
    state1.occupations[j] -= 1
    state1.coefficient = state2.coefficient * sqrt(n)
    state1.iszero = (state1.coefficient == 0.0)

    return state1
end

function ad_j!(state1::MutableFockState, state2::MutableFockState, j::Int)
    n = state2.occupations[j]
    cutoff = state2.space.cutoff

    if n == cutoff
        # Out of range → zero state
        state1.coefficient = 0.0
        state1.iszero = true
        return state1
    end

    # Copy occupations from state2 → state1
    @inbounds for k in eachindex(state2.occupations)
        state1.occupations[k] = state2.occupations[k]
    end

    # Apply a_j†
    state1.occupations[j] += 1
    state1.coefficient = state2.coefficient * sqrt(n + 1)
    state1.iszero = (state1.coefficient == 0.0)

    return state1
end


###################### VectorInterface.jl compatibility for MultipleFockState ################

mutable struct MutableFockVector
    basis::Dict{UInt32, UInt16}
    vector::Vector{MutableFockState}
end



function MutableFockVector(states::Vector{MutableFockState})
    @assert !isempty(states) "Need at least one state"

    cutoff = states[1].space.cutoff   # assume identical space

    basis = Dict{UInt32, UInt64}()
    vector = Vector{MutableFockState}(undef, length(states))

    @inbounds for i in eachindex(states)
        st = states[i]
        key = key_from_occup(UInt8.(st.occupations), cutoff)
        basis[key] = UInt32(i)             # store index
        vector[i] = st             # store the state
    end

    return MutableFockVector(basis, vector)
end

function to_fock_state(v::MutableFockVector)
    return MultipleFockState(to_fock_state.(v.vector))
end

function Base.copy(fv::MutableFockVector)
    MutableFockVector(
        copy(fv.basis),                    # copy the Dict so the new one is independent
        [copy(s) for s in fv.vector]      # copy each MutableFockState
    )
end

function LinearAlgebra.dot(x::MutableFockVector, y::MutableFockVector) 
    sum = zero(ComplexF64)
    @inbounds for i in values(x.basis)
        (x.vector[i].iszero && y.vector[i].iszero) && continue 
        sum += conj(x.vector[i].coefficient) * y.vector[i].coefficient 
    end
    return sum
end



##############################
# Scalar type
##############################

VectorInterface.scalartype(::MutableFockVector) = ComplexF64

##############################
# Zero vectors
##############################


# In-place zero for MultipleMutableFockState
function VectorInterface.zerovector!(x::MutableFockVector)
    for v in x.vector
        v.coefficient = zero(ComplexF64)
        v.iszero = true
    end
    return x
end

# BangBang for container: just call in-place if mutable
VectorInterface.zerovector!!(v::MutableFockVector) = VectorInterface.zerovector!(v)

# Out-of-place zero
function VectorInterface.zerovector(v::MutableFockVector)
    return zerovector!(copy(v))
end

##############################
# Scaling
##############################

function VectorInterface.scale!(v::MutableFockVector, α)
    for s in v.vector
        if s.iszero 
            continue
        elseif α==zero(ComplexF64)
            s.iszero=true
            s.coefficient = zero(ComplexF64)
        else
            s.coefficient *= ComplexF64(α)
        end
    end
    return v
end

VectorInterface.scale!!(s::MutableFockVector, α) = VectorInterface.scale!(s, α)
VectorInterface.scale(s::MutableFockVector, α) = VectorInterface.scale!(copy(s), α)

    

##############################
# Addition
##############################

function VectorInterface.add!(w::MutableFockVector, v::MutableFockVector; α=1, β=1)
    for i in eachindex(w.vector)
        w_i, v_i = w.vector[i], v.vector[i]
        w_i.coefficient = β * w_i.coefficient + α * v_i.coefficient
        w_i.iszero = (w_i.coefficient == 0)
    end
    return w
end

VectorInterface.add!!(w::MutableFockVector, v::MutableFockVector, α=1, β=1) =
    VectorInterface.add!(w, v; α=α, β=β)

VectorInterface.add(w::MutableFockVector, v::MutableFockVector, α=1, β=1) =
    VectorInterface.add!(copy(w), v; α=α, β=β)

##############################
# Inner product
##############################

function VectorInterface.inner(v::MutableFockVector, w::MutableFockVector)
    s = zero(ComplexF64)
    for i in eachindex(v.vector)
        v.vector[i].iszero && continue
        s += v.vector[i].coefficient * conj(w.vector[i].coefficient)
    end
    return s
end

##############################
# Norm
##############################

function VectorInterface.norm(v::MutableFockVector)
    sqrt(sum(abs2(s.coefficient) for s in v.vector))
end





