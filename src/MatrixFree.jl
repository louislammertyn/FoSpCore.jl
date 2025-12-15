struct MatrixFreeOperator
    transitions::Vector{Vector{Tuple{Int, ComplexF64}}}
end

function transition_representation(Op::MultipleFockOperator, basis::Vector{AbstractFockState})
    basis_v = MutableFockVector(MutableFockState.(basis))
    buf1 = MutableFockState(basis[1])
    buf2 = MutableFockState(basis[1])

    D = length(basis)
    transitions = [Vector{Tuple{Int, ComplexF64}}() for i in 1:D]
    
    for (j,v) in enumerate(basis_v.vector) 
        trans = Vector{Vector{Any}}()
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
                idx = basis_v.basis[key_from_occup(buf1.occupations, v.space.cutoff)] 
                eidx = findfirst(e -> e[1]==idx, trans )
                if isnothing(eidx)
                    push!(trans, [idx, o_term.coefficient * buf1.coefficient])
                else 
                    trans[eidx][2] += o_term.coefficient * buf1.coefficient
                end
                
            end
        end
        transitions[j] = Tuple.(trans)
    end
    return MatrixFreeOperator(transitions)
end


function apply!(Op::MatrixFreeOperator, w::Vector, v::Vector)
    D = length(v)
    @assert ((length(v) == length(w)) && length(w) == length(Op.transitions))
    zerovector!(w)
    for i in 1:D
        vi = v[i]
        for (j, c) in Op.transitions[i]
            w[j] += c * vi
        end
    end
    return w 
end

