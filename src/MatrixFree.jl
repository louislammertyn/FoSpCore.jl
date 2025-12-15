struct MatrixFreeOperator
    transitions::Vector{Vector{Tuple{Int, ComplexF64}}}
end

function transition_representation(Op::MultipleFockOperator, basis::Vector{AbstractFockState})
end