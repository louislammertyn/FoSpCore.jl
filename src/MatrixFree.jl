struct MatrixFreeOperator <: AbstractMatrix{ComplexF64}
    transitions::Vector{Vector{Tuple{Int, ComplexF64}}}
end

# Size of the operator
Base.size(A::MatrixFreeOperator) = (length(A.transitions), length(A.transitions))
Base.copy(A::MatrixFreeOperator) = MatrixFreeOperator([copy(row) for row in A.transitions])
zeroMFO(A::MatrixFreeOperator) = MatrixFreeOperator([Vector{Tuple{Int, ComplexF64}}() for i in 1:length(A.transitions)])
zeroMFO(L::Int) = MatrixFreeOperator([Vector{Tuple{Int, ComplexF64}}() for i in 1:L])

function Base.getindex(A::MatrixFreeOperator, i::Int, j::Int)
    for (l, c) in A.transitions[i]
        if l==j 
            return c 
        end
    end
    return zero(ComplexF64)
end

function Base.setindex!(A::MatrixFreeOperator, val::Number, i::Int, j::Int)
    for (k, (l, _)) in enumerate(A.transitions[i])
        if l==j 
            if iszero(val)
                deleteat!(A.transitions[i], k)
            else
                A.transitions[k] = (j, ComplexF64(val))
            end
            return nothing
        end
    end   
    push!(A.transitions[i], (j, val))
    
end

function Base.:+(A::MatrixFreeOperator, B::MatrixFreeOperator)
    @assert size(A) == size(B)
    C = zeroMFO(A)
    for i in 1:length(A.transitions)
        row_dict = Dict{Int, ComplexF64}()
        for (ja, ca) in A.transitions[i]
            row_dict[ja] = ca
        end
        for (jb, cb) in B.transitions[i]
            row_dict[jb] = get(row_dict, jb, zero(ComplexF64)) + cb 
        end

        for (j, val) in row_dict
            push!(C.transitions[i], (j, val))
        end
    end
    return C 
end

function Base.:*(a::Number, A::MatrixFreeOperator)
    A_ = copy(A)
    for v in A_.transitions
        for (k,(j,c)) in enumerate(v)
            v[k] = (j, a*c)
        end
    end
    return A_ 
end 

Base.:*(A::MatrixFreeOperator, a::Number) = a*A



        

# Matrix-vector multiplication
function Base.:*(A::MatrixFreeOperator, x::AbstractVector)
    y = zeros(ComplexF64, length(A.transitions))
    for (i, row) in enumerate(A.transitions)
        for (j, val) in row
            y[i] += val * x[j]
        end
    end
    return y
end


# Optional: transpose (adjoint)
function Base.transpose(A::MatrixFreeOperator) 
    A_t = zeroMFO(A)
    for i in 1:size(A)[1]
        for (j, c) in A.transitions[i]
            A_t[j, i] = c 
        end
    end
    return A_t
end


function Base.adjoint(A::MatrixFreeOperator) 
     A_t = zeroMFO(A)
    for i in 1:size(A)[1]
        for (j, c) in A.transitions[i]
            A_t[j, i] = conj(c) 
        end
    end
    return A_t
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
