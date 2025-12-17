
############### In this part we define the operators acting on a Fock space ########
# Each FockOperator consists of a product of creation and annihilation operators encoded as a tuple of tuples. 
# Each tuple corresponds to an (i,a/ad) with position i and {ad <-> true, a <-> false}
# The order of the tuple corresponds to the way we read the states on paper i.e. THE LAST TUPLE WILL BE THE ONE TO ACT FIRST ON THE STATE
# There is also a coefficient for the term
# Example:  θ = c1 * ad_1 * ad_2 * a_3 * a_2 * ad_6    <->     θ = FockOperator[( (1, true), (2, true), (3, false), (2, false), (6,true) ), c1]
# MultipleFockOperator represents sums of FockOperator types
begin

abstract type AbstractFockOperator end

struct FockOperator <: AbstractFockOperator
    product::NTuple{N, Tuple{Int,Bool}} where N 
    coefficient::ComplexF64
    space::AbstractFockSpace
end

mutable struct MultipleFockOperator <: AbstractFockOperator
    terms::Vector{FockOperator}
    cnumber::ComplexF64
end

# Neutral element structure
struct ZeroFockOperator <: AbstractFockOperator
end

Base.zero(::Type{FockOperator}) = ZeroFockOperator()
Base.zero(::Type{MultipleFockOperator}) = ZeroFockOperator()


# empty product means identity

function identity_fockoperator(V::AbstractFockSpace, c::ComplexF64=1.0)
    FockOperator(NTuple{0, Tuple{Int,Bool}}() , c, V) 
end

Base.:+(z::ZeroFockOperator, z2::ZeroFockOperator) = z
Base.:-(z::ZeroFockOperator, z2::ZeroFockOperator) = z
Base.:+(z::ZeroFockOperator, c::Number) = z
Base.:+(c::Number, z::ZeroFockOperator) = z
Base.:+(z::ZeroFockOperator, s::FockOperator) = s
Base.:+(s::FockOperator, z::ZeroFockOperator) = s
Base.:+(z::ZeroFockOperator, ms::MultipleFockOperator) = ms
Base.:+(ms::MultipleFockOperator, z::ZeroFockOperator) = ms
Base.:-(z::ZeroFockOperator, c::Number) = z
Base.:-(c::Number, z::ZeroFockOperator) = z
Base.:-(z::ZeroFockOperator, s::FockOperator) = -1 *s
Base.:-(s::FockOperator, z::ZeroFockOperator) = s
Base.:-(z::ZeroFockOperator, ms::MultipleFockOperator) =-1* ms
Base.:-(ms::MultipleFockOperator, z::ZeroFockOperator) = ms
Base.:*(c::Number, z::ZeroFockOperator) = z
Base.:*(z::ZeroFockOperator, c::Number) = z
Base.:*(z1::ZeroFockOperator, z2::ZeroFockOperator) = 0
Base.:*(z::ZeroFockOperator, s::FockOperator) = 0
Base.:*(s::FockOperator, z::ZeroFockOperator) = 0

############ Pretty Printing ############
function Base.show(io::IO, op::FockOperator)
    str = op.coefficient == 1 + 0im ? "" : string("($(op.coefficient))", " ⋅ ")

    for (site, is_creation) in op.product
        str *= is_creation ? "a†($site)" : "a($site)"
        str *= " "
    end

    print(io, strip(str))
end

function Base.show(io::IO, mop::MultipleFockOperator)
    terms_empty = isempty(mop.terms)
    has_cnumber = mop.cnumber != 0

    if terms_empty && !has_cnumber
        print(io, "0")
        return
    end

    shown_any = false

    # print cnumber first if it exists
    if has_cnumber
        str = mop.cnumber == 1+0im ? "𝟙" : "($(mop.cnumber)) ⋅ 𝟙"
        print(io, str)
        shown_any = true
    end

    # print operator terms
    for term in mop.terms
        if shown_any
            print(io, " + ")
        end
        print(io, term)
        shown_any = true
    end
end

########## Basic operations ##########
Base.size(Op::FockOperator) = (prod(Op.space.geometry), prod(Op.space.geometry))

Base.size(Op::MultipleFockOperator) = size(Op.terms[1])

Base.eltype(Op::AbstractFockOperator) = ComplexF64

Base.:+(op1::FockOperator, op2::FockOperator) =
    op1.product == op2.product ? cleanup_FO(FockOperator(op1.product, op1.coefficient + op2.coefficient, op1.space)) : MultipleFockOperator([op1, op2], 0. + 0im);

Base.:-(op1::FockOperator, op2::FockOperator) = op1 + FockOperator(op2.product, -op2.coefficient, op2.space)

function Base.:+(op::FockOperator, mop::MultipleFockOperator)
    new_terms = copy(mop.terms)
    matched = false
    for i in eachindex(new_terms)
        if new_terms[i].product == op.product
            new_terms[i] = FockOperator(op.product, new_terms[i].coefficient + op.coefficient, op.space)
            matched = true
            break
        end
    end
    if !matched
        push!(new_terms, op)
    end
    
    return cleanup_FO(MultipleFockOperator(new_terms, mop.cnumber))
end

Base.:+(mop::MultipleFockOperator, c::Number) = (mop.cnumber += c; return mop)
Base.:-(mop::MultipleFockOperator, c::Number) =(mop.cnumber -= c; return mop)
Base.:+(c::Number, mop::MultipleFockOperator) = (mop.cnumber += c; return mop)
Base.:-(c::Number, mop::MultipleFockOperator) = (mop.cnumber *= -1; mop.cnumber += c; return mop)

Base.:+(mop::MultipleFockOperator, op::FockOperator) = op + mop
Base.:-(mop::MultipleFockOperator, op::FockOperator) = mop + FockOperator(op.product, -op.coefficient, op.space)
Base.:-(op::FockOperator, mop::MultipleFockOperator) = (-1) * mop + op


function Base.:+(mop1::MultipleFockOperator, mop2::MultipleFockOperator)
    result = copy(mop2)
    for t in mop1.terms
        result = result + t
    end
    result += mop1.cnumber
    return cleanup_FO(result)
end

Base.:-(mop1::MultipleFockOperator, mop2::MultipleFockOperator) = mop1 + (-1) * mop2

Base.:*(c::Number, op::FockOperator) = FockOperator(op.product, c * op.coefficient, op.space)
Base.:*(op::FockOperator, c::Number) = c * op

function Base.:*(c::Number, mop::MultipleFockOperator)
    new_terms = [c * t for t in mop.terms]
    return cleanup_FO(MultipleFockOperator(new_terms, c*mop.cnumber))
end

Base.:*(mop::MultipleFockOperator, c::Number) = c * mop

# Multiplying operators
function Base.:*(Op1::FockOperator, Op2::FockOperator)
    factors1 = collect(Op1.product)
    factors2 = collect(Op2.product)
    new_factor = vcat(factors1, factors2)
    return FockOperator(Tuple(new_factor), Op1.coefficient * Op2.coefficient, Op1.space)
end

function Base.:*(MOp::MultipleFockOperator, Op::FockOperator)
    terms = Vector{FockOperator}()
    for O in MOp.terms
        push!(terms, O * Op)
    end
    push!(terms, Op * MOp.cnumber)
    return MultipleFockOperator(terms, 0.)
end

function Base.:*(Op::FockOperator, MOp::MultipleFockOperator)
    terms = Vector{FockOperator}()
    for O in MOp.terms
        push!(terms, Op * O)
    end
    push!(terms, Op * MOp.cnumber)
    return MultipleFockOperator(terms, 0.)
end

function Base.:*(MOp1::MultipleFockOperator, MOp2::MultipleFockOperator)
    terms = Vector{FockOperator}()
    for O1 in MOp1.terms, O2 in MOp2.terms
        push!(terms, O1 * O2)
        push!(terms, O1 * MOp2.cnumber)
        push!(terms, O2 * MOp1.cnumber)
    end
    
    return MultipleFockOperator(terms, MOp1.cnumber * MOp2.cnumber)
end

########## Utilities ##########
function Base.copy(op::FockOperator)
    return FockOperator(op.product, op.coefficient, op.space)
end

Base.copy(mop::MultipleFockOperator) = MultipleFockOperator(copy(mop.terms), copy(mop.cnumber))

function cleanup_FO(op::FockOperator)
    return op.coefficient==0. ? ZeroFockOperator() : op
end
function cleanup_FO(mop::MultipleFockOperator)
    new_terms = filter(t -> !isapprox(abs2(t.coefficient), 0; atol=1e-15), mop.terms)
    if length(new_terms) == 0
        return ZeroFockOperator()
    else
        return MultipleFockOperator(new_terms, mop.cnumber)
    end
end

function cleanup_FO(mop::ZeroFockOperator)
    return mop
end

function dagger_FO(Op::FockOperator)
    c_dag = Op.coefficient'
    new_terms = []
    for o in reverse(Op.product)
        tup = (o[1],!o[2])
        push!(new_terms, tup)
    end
    return FockOperator(Tuple(new_terms), c_dag, Op.space)
end

function dagger_FO(Ops::MultipleFockOperator)
    new_ops = []
    for op in Ops.terms
        push!(new_ops, dagger_FO(op))
    end
    return MultipleFockOperator(new_ops, Ops.cnumber')
end

function Base.:*(Op::FockOperator, ket::AbstractFockState)
    for factor in reverse(Op.product)
        if factor[2]
            ket = ad_j(ket, factor[1])
        else
            ket = a_j(ket, factor[1])
        end
    end 
    return Op.coefficient * ket
end

function Base.:*(Ops::MultipleFockOperator, ket::AbstractFockState)
    new_ket = ZeroFockState()

    for Op in Ops.terms
        new_ket = new_ket + (Op * ket)
        checkU1(new_ket)
    end
    new_ket += Ops.cnumber * ket
    return new_ket
end 

function apply(Op::MultipleFockOperator, ket::MultipleFockState)::MultipleFockState
    return Op * ket
end


function apply!(Op::FockOperator, ket::MutableFockState)
    for (site, ladder) in reverse(Op.product)
        ladder ? ad_j!(ket, site) : a_j!(ket, site)
        ket.iszero && return nothing 
    end
    mul_Mutable!(Op.coefficient, ket)
end    

function apply!(Op::MultipleFockOperator, w::MutableFockVector, V::MutableFockVector)
    buf1 = deepcopy(V.vector[1])
    buf2 = deepcopy(V.vector[1])
    zerovector!(w)
    for v in V.vector 
        for o_term in Op.terms 
            op_str = o_term.product
            L = length(op_str)
            if op_str[L][2]
                ad_j!(buf1, v, op_str[L][1])
            else
                a_j!(buf1, v, op_str[L][1])
            end
            for i in (L-1):-1:1
                if op_str[i][2]
                    ad_j!(buf2, buf1, op_str[i][1])
                else
                    a_j!(buf2, buf1, op_str[i][1])
                end
                
                buf1.occupations .= buf2.occupations                
                buf1.coefficient = buf2.coefficient
                buf1.iszero = buf2.iszero

                if buf1.iszero
                    break
                end
            end
            if buf1.iszero
                continue
            else
                idx = w.basis[key_from_occup(buf1.occupations, v.space.cutoff)] 
                w.vector[idx].coefficient += o_term.coefficient * buf1.coefficient
                w.vector[idx].iszero = false
            end
        end
    end
    return w
end

function key_from_occup(occup::Vector{UInt8}, cutoff::Int)::UInt64
    if cutoff > 1

        key = UInt64(0)
        factor = UInt64(1)
        factor_cutoff = UInt64(cutoff + 1)
        @assert factor_cutoff^(length(occup)) < 2e62
        @inbounds for n in occup
            key += UInt64(n) * factor
            factor *= factor_cutoff
        end
        return key
    else
        key = 0
        particles = sum(occup)
        M= length(occup)
        @assert binomial(M, particles) < 2e62
        for (i, bit) in enumerate(occup)
            if isone(bit)
                key += binomial(M-i, particles)
                particles -= 1
            end
        end
        return UInt64(key)
    end
end




end;
