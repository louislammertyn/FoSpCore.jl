begin
abstract type AbstractFockString end

struct SameSiteString <: AbstractFockString
    bits::UInt64 
    len::Int
    function SameSiteString(bits::UInt64, len::Int)
        @assert len <= 64
        mask = UInt64(1) << len - 1
        new(bits & mask, len)
    end
end

struct MultiSiteString <: AbstractFockString
    factors::Dict{Int, SameSiteString}
end

function Base.show(io::IO, s::SameSiteString)
    print(bitstring(s.bits)[end-(s.len-1) : end] )
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
    bits, len = remove_two_bits(s.bits, s.len, min(i, j), max(i,j))
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

function check_normal_ordering(s::SameSiteString)
    bs = s.bits
    bs_neg = flip_bits_range(bs, 0, s.len)
    return leading_zeros(bs_neg) == 64 - count_ones(bs_neg)
end

function normal_order(s::SameSiteString)
    results = Dict{SameSiteString, Int}()
    cache = [s]

    while !isempty(cache)
        current = pop!(cache)  # take last element (stack)
        
        if check_normal_ordering(current)
            results[current] = get(results, current, 0) + 1
        else
            contracted, commuted = commute_first_pair(current)      
            push!(cache, contracted, commuted)
        end
        
    end

    return results     
end


function group_sites_to_strings(Ops::NTuple{N, Tuple{Int, Bool}}) where N
    bits  = Dict{Int, UInt64}()
    lens  = Dict{Int, Int}()

    for j in eachindex(Ops)
        site, is_creation = Ops[N - j+1]
        i = get!(lens, site, 0)
        is_creation && (bits[site] = get(bits, site, 0) | (UInt64(1) << i))
        lens[site] = i + 1
        
    end
    

    return Dict(site => SameSiteString(get(bits,site,UInt64(0)), lens[site]) for site in keys(lens))
end

function expand_site(site::Int, s::SameSiteString,
                             cre::Vector{Tuple{Int,Bool}},
                             ann::Vector{Tuple{Int,Bool}})
    bits = s.bits
    len  = s.len
    @inbounds for i in 1:len
        if (bits >> (i-1)) & 0x1 == 1
            push!(cre, (site, true))
        else
            push!(ann, (site, false))
        end
    end
end


function normal_order(O::NTuple{N,Tuple{Int, Bool}}, c::ComplexF64, V::AbstractFockSpace) where N
    is_normal_ordered(O) && return FockOperator(O, c, V)

    # 1. group operators by site → SameSiteString
    site_strings = group_sites_to_strings(O)

    # 2. normal order each site
    ordered_per_site = Dict{Int, Dict{SameSiteString, Int}}()
    for (site, sss) in site_strings
        ordered_per_site[site] = normal_order(sss)
    end

    # 3. Cartesian product over sites
    sites = sort(collect(keys(ordered_per_site)))
    site_expansions = [ordered_per_site[s] for s in sites]

    result = MultipleFockOperator([], 0)
    for combo in Iterators.product(site_expansions...)
        coeff = c
        creation = Tuple{Int,Bool}[]
        annihilation = Tuple{Int,Bool}[]

        for ((sss, count), site) in zip(combo, sites)
            coeff *= count
            expand_site(site, sss, creation, annihilation)
        end

        if isempty(creation) && isempty(annihilation)
            if typeof(result) == FockOperator
                result = MultipleFockOperator([result], 0)
            end
            result.cnumber += coeff
        else
            full_ops = Tuple(vcat(creation, annihilation))
            result += FockOperator(full_ops, coeff, V)
        end
    end

    return typeof(result) == FockOperator ? result : remove_zeros(result)
end


function commutator(O1::AbstractFockOperator, O2::AbstractFockOperator)
    return O1 * O2 - O2 * O1
end

end;
