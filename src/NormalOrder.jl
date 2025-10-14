
begin

abstract type AbstractFockString end

mutable struct SameSiteString <: AbstractFockString
    factors::Vector{Bool} 
end

mutable struct MultiSiteString <: AbstractFockString
    factors::Dict{Int, SameSiteString}
end

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
    new_Os.cnumber = Os.cnumber
    return new_Os
end

function commutator(O1::AbstractFockOperator, O2::AbstractFockOperator)
    return normal_order(O1 * O2) - normal_order(O2 * O1)
end

end;