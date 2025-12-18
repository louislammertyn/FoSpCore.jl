
abstract type AbstractFockString end

struct SameSiteString <: AbstractFockString
    bits::UInt64 
    len::Int
end

struct MultiSiteString <: AbstractFockString
    factors::Dict{Int, SameSiteString}
end

function flip_bits_range(bits::UInt64, i::Int, j::Int)
    # i = leftmost position to flip
    # j = rightmost position to flip
    @assert i <= j <= 64
    mask = ((UInt64(1) << (j - i + 1)) - 1) << (i - 1)
    return bits ⊻ mask
end

function first_ann_cre_pair(bits::UInt64, len::Int)::Int
    neg_bits = flip_bits_range(bits, 1, len)
    shifted = neg_bits >> 1
    candidates = bits & shifted 
    return 64 - leading_zeros(candidates) + 1
end

function remove_two_bits(bits::UInt64, len::Int, i::Int, j::Int)
    @assert i < j <= len

    # Lower part: bits below i
    lower = bits & ((UInt64(1) << (i-1)) - 1)

    # Middle part: bits between i and j
    middle = (bits >> i) & ((UInt64(1) << (j - i - 1)) - 1)

    # Upper part: bits above j
    upper = bits >> (j)

    # Combine: shift middle and upper down to fill the gaps
    new_bits = lower | (middle << (i-1)) | (upper << (j-2))
    new_len = len - 2

    return new_bits, new_len
end

function remove_pair(s::SameSiteString, i::Int, j::Int)
    bits, len = remove_two_bits(s.bits, s.len, i, j)
    return SameSiteString(bits, len)
end

function first_ann_cre_pair(s::SameSiteString)
    id_ann = first_ann_cre_pair(s.bits, s.len) 
    id_cr = id_ann - 1
    return id_ann, id_cr
end

function switch_pair(s::SameSiteString, id_ann::Int)
    switched = flip_bits_range(s.bits, id_ann-1, id_ann)
    return SameSiteString(switched, s.len)
end

function commute_first_pair(s::SameSiteString)
    id_ann, id_cr = first_ann_cre_pair(s)
    contracted = remove_pair(s, id_cr, id_ann)
    commuted = switch_pair(s, id_ann)
    return contracted, commuted
end
bitstring(UInt64(9))
s = SameSiteString(UInt64(9), 5)
for i in 1:4
    contracted, commuted = commute_first_pair(s)
    #println(contracted.len)
    #println(commuted.len)
    #println(bitstring(contracted.bits))
    println(bitstring(commuted.bits))
    s = commuted
end
first_ann_cre_pair(s)

function commute_first!(ops::SameSiteString)
    indicator = true
    for (i,a) in enumerate(ops.factors)
        if !indicator && a
            id_term = deepcopy(ops)
            deleteat!(id_term.factors, [i-1,i])
            ops.factors[i-1] = true
            ops.factors[i] = false
            
            return ops, id_term
        end
        indicator = a
    end
    return ops, false
end

function normal_order!(ops::SameSiteString)
    result = Dict{Vector{Bool}, Int}()
    _normal_order!(deepcopy(ops), result)
    return result
end

function _normal_order!(ops::SameSiteString, acc::Dict{Vector{Bool}, Int})
    # If already normal ordered (creations then annihilations), store it
    if issorted(ops.factors; rev=true)
        acc[ops.factors] = get(acc, ops.factors, 0) + 1
        return nothing
    end

    # Try to commute first out-of-order pair
    new_ops = deepcopy(ops)
    commuted, id_term = commute_first!(new_ops)

    # Recurse on commuted term
    _normal_order!(deepcopy(commuted), acc)

    # Recurse on identity term (commutator) if it exists
    if id_term != false
        _normal_order!(id_term, acc)
    end
end

function normal_order(O::FockOperator)
    c = O.coefficient

    # Group operators by site
    site_dict = Dict{Int, Vector{Bool}}()
    for (i, b) in O.product
        push!(get!(site_dict, i, Bool[]), b)
    end

    # Normal order each site
    ordered_per_site = Dict{Int, Dict{Vector{Bool}, Int64}}()
    for (site, bools) in site_dict
        str = SameSiteString(copy(bools))
        ordered_per_site[site] = normal_order!(str)
    end

    # Cartesian product over sites
    sites = sort(collect(keys(ordered_per_site)))
    site_orderings = [ordered_per_site[site] for site in sites]

    # Combine all ordered strings
    result = MultipleFockOperator([], 0)

    for combination in Iterators.product(site_orderings...)
        coeff_factor = c
        creation_part = Tuple{Int, Bool}[]
        annihilation_part = Tuple{Int, Bool}[]

        for (site_idx, (ops_dict, site)) in enumerate(zip(combination, sites))
            ops, count = ops_dict
            coeff_factor *= count
            for b in ops
                if b
                    push!(creation_part, (site, true))
                else
                    push!(annihilation_part, (site, false))
                end
            end
        end

        if isempty(creation_part) && isempty(annihilation_part)
            println(coeff_factor)
            println(result)
            result.cnumber += coeff_factor
            println(result)
        else
            full_ops = vcat(creation_part, annihilation_part)
            result += FockOperator(Tuple(full_ops), coeff_factor, O.space)
        end
    end

    return result
    
end

function normal_order(Os::MultipleFockOperator)
    new_Os = ZeroFockOperator()
    for o in Os.terms
        new_Os += normal_order(o)
    end
    typeof(new_Os) == ZeroFockOperator && return new_Os
    new_Os.cnumber = Os.cnumber
    return new_Os
end

function commutator(O1::AbstractFockOperator, O2::AbstractFockOperator)
    return normal_order(O1 * O2) - normal_order(O2 * O1)
end

