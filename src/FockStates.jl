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

# Equality
import Base: ==
==(f1::U1FockSpace{D}, f2::U1FockSpace{D}) where D =
    f1.geometry == f2.geometry &&
    f1.cutoff == f2.cutoff &&
    f1.particle_number == f2.particle_number

==(f1::UnrestrictedFockSpace{D}, f2::UnrestrictedFockSpace{D}) where D =
    f1.geometry == f2.geometry &&
    f1.cutoff == f2.cutoff

#U1FockSpace(geometry::NTuple{D, Int}, cutoff::Int, particle_number::Int) where D = U1FockSpace{D}(geometry, cutoff, particle_number)

dimension(ufs::U1FockSpace) = prod(ufs.geometry) * (min(ufs.cutoff, ufs.particle_number))
dimension(ufs::UnrestrictedFockSpace) = prod(ufs.geometry) * ufs.cutoff

single_particle_space(geometry::NTuple{D,Int}) where D = U1FockSpace(geometry, 1, 1)

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



==(s1::FockState, s2::FockState) =
    s1.occupations == s2.occupations &&
    s1.coefficient == s2.coefficient &&
    s1.space == s2.space

==(m1::MultipleFockState, m2::MultipleFockState) =
    length(m1.states) == length(m2.states) &&
    all(m1.states .== m2.states)  # element-wise equality

==(z1::ZeroFockState, z2::ZeroFockState) = true
==(z::ZeroFockState, x) = x isa ZeroFockState  # ensures symmetric equality


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
    if typeof(fs) == U1FockSpace
        @assert sum(occs) == fs.particle_number "occupation does not match given particle number"
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
    if typeof(fs) == U1FockSpace
        @assert sum(occs) == fs.particle_number "occupation does not match given particle number"
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

function remove_zeros(mstates::MultipleFockState)
    new_states = filter(s -> !(isapprox(norm(s.coefficient) , 0;atol=1e-12)), mstates.states)    
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
        return fock_state(state.space, occs, 0.)
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


###################### Krylov Method functions #########################
function rand_superpos(basis::Vector{AbstractFockState})
    n_basis = rand(ComplexF64, length(basis)) .* basis |> MultipleFockState
    norm = n_basis |> norm2FS |> sqrt
    return  n_basis * (1/norm)
end



###################### Generating states ############################
function all_states_U1(V::U1FockSpace) 
    
    N= V.particle_number
    L = prod(V.geometry)
    U1occs = bounded_compositions(N, L, V.cutoff)
    states = Vector{AbstractFockState}()
    for occ in U1occs
        push!(states, fock_state(V, occ))
    end
    return states
end

function all_states_U1( V::UnrestrictedFockSpace)
    states = []
    ranges = ntuple(_->0:V.cutoff, prod(V.geometry))
    U1occs = [collect(t) for t in Iterators.product(ranges...)]
    println(U1occs)
    for occ in U1occs
        push!(states, fock_state(V, occ))
    end
    return states
end

function all_states_U1_O(V::U1FockSpace) 
    geometry = V.geometry
    L = prod(geometry)
    N = V.particle_number

    # initial Fock state
    v_i = zeros(Int, L)
    v_i[1] = N
    fs_i = fock_state(V, v_i)

    # nearest-neighbor hopping
    T = ZeroFockOperator()
    for i in 1:(L-1)
        T += FockOperator(((i,true),(i+1,false)), one(ComplexF64), V)
    end
    T += dagger_FO(T)

    # iterative BFS-like exploration
    queue = [fs_i]
    visited = Set([fs_i.occupations])

    while !isempty(queue)
        s = pop!(queue)
        ns = T * s
        if typeof(ns)==FockState
            ns = MultipleFockState([ns])
        end
        for s_new in ns.states
            if  !(s_new.occupations in visited)
                push!(visited, s_new.occupations)
                push!(queue, s_new)
            end
        end
    end
    result::Vector{AbstractFockState} = [fock_state(V, collect(occs)) for occs in visited]
    sorted = sort(result, by=x -> Tuple(x.occupations), rev=true)
    return sorted
end


function bounded_compositions(N::Int, L::Int, cutoff::Int; thread_threshold::Int=10_000)
    cutoff += 1
    max_i = cutoff^L  
    nthreads = Threads.nthreads()
    
    if max_i < thread_threshold || Threads.nthreads() == 1
        # ---------------- Single-threaded version ----------------
        results = Vector{Vector{Int}}()
        for i in 0:max_i-1
            n = digits(i, base=cutoff)
            if sum(n) == N && length(n) <= L
                push!(results, reverse(n))
            end
        end
    else
        # ---------------- Multithreaded version ----------------
        thread_results = [Vector{Vector{Int}}() for _ in 1:nthreads]
        blocksize = ceil(Int, max_i / nthreads)

        Threads.@threads for t in 1:nthreads
            start_i = (t-1) * blocksize
            stop_i = min(t * blocksize-1, max_i-1)
            
            for i in start_i:stop_i
                n = digits(i, base=cutoff)
                if sum(n) == N && length(n) <= L
                    push!(thread_results[t], reverse(n))
                end
            end
        end
        results = reduce(vcat, thread_results)
    end

    # Padding & sorting (shared by both paths)
    padded = results .|> x -> vcat(zeros(Int, L - length(x)), x)
    sorted = sort(padded, by=x -> Tuple(x), rev=true)
    return sorted
end


function basisFS(space::U1FockSpace; nodata=true, savedata=false)
    dirpath = "./src/assets/states"
    savename = "basis_u1_geom=$(join(space.geometry, 'x'))_cutoff=$(space.cutoff)_N=$(space.particle_number).jld2"
    savepath = joinpath(dirpath, savename)

    (nodata & !savedata) && return all_states_U1_O(space)
    
    # Create directory if it doesn't exist
    if !isdir(dirpath)
        mkpath(dirpath)
    end

    # Case 1: File exists and we want to use it
    if isfile(savepath) && !nodata
        data = load(savepath)
        return data["states"]

    # Case 2: File doesn't exist, but we want to save new data
    elseif !isfile(savepath) && savedata
        states = all_states_U1(space)
        save(savepath, Dict("states" => states))
        return states

    # Case 3: We don’t want to use or save data (pure computation)
    else
        return all_states_U1_O(space)
    end
end


end;
