begin

########## Definition of objects ##########
## We take two types of spaces 
# The state satisfies U(1) symmetry or not 

## We define two types of states belonging to the same abstract type for fock states
# The single state and the sum of states as a set of single states
abstract type AbstractFockSpace end

struct U1FockSpace{D} <: AbstractFockSpace
    geometry::NTuple{D, Int}
    cutoff::Int
    particle_number::Int
    function U1FockSpace(geometry::NTuple{D, Int}, cutoff::Int, particle_number::Int) where D
        # you can add validation or preprocessing here
        # For example:
        if any(x -> x <= 0, geometry)
            error("All geometry dimensions must be positive")
        end
        new{D}(geometry, cutoff, particle_number)
    end
end

struct UnrestrictedFockSpace{D} <: AbstractFockSpace
    geometry::NTuple{D, Int}
    cutoff::Int
  
end

#U1FockSpace(geometry::NTuple{D, Int}, cutoff::Int, particle_number::Int) where D = U1FockSpace{D}(geometry, cutoff, particle_number)

dimension(ufs::U1FockSpace) = prod(ufs.geometry) * (min(ufs.cutoff, ufs.particle_number))
dimension(ufs::UnrestrictedFockSpace) = prod(ufs.geometry) * ufs.cutoff



abstract type AbstractFockState end

struct FockState <: AbstractFockState
    occupations::NTuple{N,Int} where N
    coefficient::ComplexF64  # default: 1.0 + 0im
    space::AbstractFockSpace
end

mutable struct MultipleFockState <: AbstractFockState
    states::Vector{FockState}
end

struct ZeroFockState <: AbstractFockState
end

# We add a seperate Mutable FockState for performance critical codes see section 6. of this file for functionalities
mutable struct MutableFockState <: AbstractFockState
    occupations::Vector{Int}
    coefficient::ComplexF64
    space::AbstractFockSpace
    iszero::Bool
end

########## 0. Pretty printing ###########
# Single Fock state
function Base.show(io::IO, s::FockState)
    occ_str = join(s.occupations, ", ")
    coeff_str = s.coefficient == 1 + 0im ? "" : string("($(s.coefficient))", " ⋅ ")
    print(io, coeff_str * "|", occ_str, "⟩")
end

# Multiple Fock states (sum)
function Base.show(io::IO, ms::MultipleFockState)
    if isempty(ms.states)
        print(io, "0")
        return
    end

    for (i, st) in enumerate(ms.states)
        if i > 1
            print(io, " + ")
        end
        print(io, st)
    end
end

# Zero state (vacuum)
function Base.show(io::IO, ::ZeroFockState)
    print(io, "|0⟩")
end

# MutableFockState (similar to FockState)
function Base.show(io::IO, s::MutableFockState)
    occ_str = join(s.occupations, ", ")
    coeff_str = s.coefficient == 1 + 0im ? "" : string("($(s.coefficient))", " ⋅ ")
    print(io, coeff_str * "| ", occ_str, " ⟩")
end

########## 1. Basic operations ##########
# we start with the neutral element
Base.zero(s::AbstractFockState) = ZeroFockState()
Base.:+(z::ZeroFockState,z2::ZeroFockState)=z
Base.:+(z::ZeroFockState, s::FockState) = s
Base.:+(s::FockState, z::ZeroFockState) = s
Base.:+(z::ZeroFockState, ms::MultipleFockState) = ms
Base.:+(ms::MultipleFockState, z::ZeroFockState) = ms
Base.:-(z::ZeroFockState, s::FockState) = s
Base.:-(s::FockState, z::ZeroFockState) = s
Base.:-(z::ZeroFockState, ms::MultipleFockState) = ms
Base.:-(ms::MultipleFockState, z::ZeroFockState) = ms
Base.:*(c::Number, z::ZeroFockState) = z
Base.:*(z::ZeroFockState, c::Number) = z
Base.:*(z1::ZeroFockState, z2::ZeroFockState) = 0
Base.:*(z::ZeroFockState, s::FockState) = 0
Base.:*(s::FockState, z::ZeroFockState) = 0

# Note that we take the *-operation  as the inproduct: F1 * F2 = <F1|F2>; F1 x F2 -> ComplexF64
# The remaining operations correspond to our intuition


function Base.:+(state1::FockState, state2::FockState)
    if state1.occupations == state2.occupations
        return cleanup_FS(FockState(state1.occupations, state1.coefficient + state2.coefficient, state1.space))
    else
        return cleanup_FS(MultipleFockState([state1, state2]))
    end
end

function Base.:-(state1::FockState, state2::FockState)
    return state1 + FockState(state2.occupations, -state2.coefficient, state2.space)
end

function Base.:+(state::FockState, mstate::MultipleFockState)
    new_states = copy(mstate.states)
    matched = false
    for i in eachindex(new_states)
        if new_states[i].occupations == state.occupations
            new_states[i] = FockState(new_states[i].occupations, new_states[i].coefficient + state.coefficient, new_states[i].space)
            matched = true
            break
        end
    end
    if !matched
        push!(new_states, state)
    end
    return cleanup_FS(MultipleFockState(new_states))
end

Base.:+(mstate::MultipleFockState, state::FockState) = state + mstate

function Base.:-(mstate::MultipleFockState, state::FockState)
    return mstate + FockState(state.occupations, -state.coefficient, state.space)
end

Base.:-(state::FockState, mstate::MultipleFockState) = -(mstate) + state

function Base.:+(mstate1::MultipleFockState, mstate2::MultipleFockState)
    result = copy(mstate2)
    for s in mstate1.states
        result = s + result
    end
    return cleanup_FS(result)
end

function Base.:-(mstate1::MultipleFockState, mstate2::MultipleFockState)
    return mstate1 + (-1) * mstate2
end

function Base.:*(c::Number, state::FockState)
    return FockState(state.occupations, c * state.coefficient, state.space)
end

Base.:*(state::FockState, c::Number) = c * state

Base.:*(state1::FockState, state2::FockState) = (state1.occupations == state2.occupations) ? state1.coefficient' * state2.coefficient : Complex(0)

LinearAlgebra.dot(state1::FockState, state2::FockState)::ComplexF64 = state1 * state2

function Base.:*(c::Number, mstate::MultipleFockState)
    new_states = [FockState(s.occupations, c * s.coefficient, s.space) for s in mstate.states]
    return cleanup_FS(MultipleFockState(new_states))
end

Base.:*(mstate::MultipleFockState, c::Number) = c * mstate

function Base.:*(state::FockState, mstate::MultipleFockState)
    c = zero(ComplexF64)
    for s in mstate.states
        if s.occupations == state.occupations
            c += state * s
        end
    end
    return c
end

Base.:*(mstate::MultipleFockState, state::FockState) = state * mstate

function Base.:*(mstate1::MultipleFockState, mstate2::MultipleFockState)
    c = zero(ComplexF64)
    for s1 in mstate1.states
        c += s1 * mstate2
    end
    return c
end

LinearAlgebra.dot(ms1::MultipleFockState, ms2::MultipleFockState)::ComplexF64 = ms1 * ms2

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

######### 2. Basic states instantiation and functionalities ########
# Create a multi-mode basis state |n₁, n₂, ..., n_N⟩

function fock_state(fs::AbstractFockSpace, occs::Vector{Int}, coeff::Number=1. +0im)
    if length(occs) != prod(fs.geometry)
        error("Occupations must match number of modes")
    end
    for n in occs
        if n < 0 || n > fs.cutoff
            error("Occupation number out of bounds")
        end
    end
    
    return FockState(ntuple(i -> occs[i], length(occs)), ComplexF64(coeff), fs)
end

function fock_state(fs::AbstractFockSpace, occs::Array, coeff::Number=1. +0im)
    if prod(size(occs)) != prod(fs.geometry)
        error("Occupations must match number of modes")
    end
    for n in occs
        if n < 0 || n > fs.cutoff
            error("Occupation number out of bounds")
        end
    end
    
    return FockState(ntuple(i -> occs[i], length(occs)), ComplexF64(coeff), fs)
end

function Base.copy(state::FockState)
    return FockState(state.occupations, state.coefficient, state.space)
end

Base.copy(mstates::MultipleFockState) = MultipleFockState(copy(mstates.states))

function cleanup_FS(state::FockState)
    return state.coefficient==0 ? ZeroFockState() : state 
end

function cleanup_FS(mstates::MultipleFockState)
    new_states = filter(s -> s.coefficient != 0, mstates.states)    
    if length(new_states) == 1
        return new_states[1]
    else
        return MultipleFockState(new_states)   
    end
end

########## Basic properties #########
function checkU1(fs::FockState)
    if fs.space == UnrestrictedFockSpace
        @warn ("The provided state is not of type U1 symmetric")
        return false
    end
    return sum(fs.occupations) == fs.space.particle_number
end
function checkU1(fs::MultipleFockState)
    for s in fs.states
        if s.space == UnrestrictedFockSpace
            @warn ("The provided state is not of type U1 symmetric")
            return false
        end
        if sum(s.occupations) != s.space.particle_number
            error("Sum of occupations is not equal to total particle number, instead sum is $(sum(s.occupations)) while PN is $(s.space.particle_number) \n The occupations giving error are $(s.occupations)")
        end
    end
    nothing
end
checkU1(z::ZeroFockState) = nothing

function norm2FS(state::FockState)
    return  abs2(state.coefficient)
end

function norm2FS(mstates::MultipleFockState)
    return mstates * mstates
end

############# 3. Creation and annihilation operations ################

function ad_j(state::FockState, j::Int)
    if j > length(state.occupations) || j < 1
        error("Mode $j lies out of bounds")
    end
    occs = collect(state.occupations)
    if occs[j] + 1 > state.space.cutoff
        return ZeroFockState()
    end
    occs[j] += 1
    coeff = state.coefficient * sqrt(occs[j])
    return fock_state(state.space, occs, coeff)
end


function a_j(state::FockState, j::Int)
    if j > length(state.occupations) || j < 1
        error("Mode $j lies out of bounds")
    end
    occs = collect(state.occupations)
    coeff = state.coefficient
    if occs[j] > 0
        coeff *= sqrt(occs[j])
        occs[j] -= 1
    else
        
        return fock_state(state.space, occs, 0.)
    end
    return fock_state(state.space, occs, coeff)
end

function ad_j(mstate::MultipleFockState, j::Int)
    states = copy(mstate.states)
    for (i,s) in enumerate(states)
        states[i] = ad_j(s, j)
    end
    return cleanup_FS(MultipleFockState(states))
end

function a_j(mstate::MultipleFockState, j::Int)
    states = copy(mstate.states)
    for (i,s) in enumerate(states)
        states[i] = a_j(s, j)
    end
    return cleanup_FS(MultipleFockState(states))
end

a_j(state::ZeroFockState ,j) = ZeroFockState() 
ad_j(state::ZeroFockState ,j) = ZeroFockState() 


function create_MFS(coefficients::Vector{ComplexF64}, states::Vector{AbstractFockState})
    @assert length(coefficients)==length(states)
    total_state = ZeroFockState()
    for (i,c) in enumerate(coefficients)
        state = states[i] * (1/states[i].coefficient)
        total_state += c * state
    end
    return cleanup_FS(total_state)
end


################### 6. MutableFockState functionalities ###################
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
    if j < 1 || j > length(state.occupations)
        error("Mode $j out of bounds")
    end
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
    if j < 1 || j > length(state.occupations)
        error("Mode $j out of bounds")
    end
    n = state.occupations[j]
    if n + 1 > state.space.cutoff
        state.coefficient = 0.0
        state.iszero = true
    else
        state.coefficient *= sqrt(n + 1)
        state.occupations[j] += 1
        
    end
end

###################### Krylov Method functions #########################
function rand_superpos(basis::Vector{AbstractFockState})
    n_basis = rand(ComplexF64, length(basis)) .* basis |> MultipleFockState
    norm = n_basis |> norm2FS |> sqrt
    return  n_basis * (1/norm)
end

end;
