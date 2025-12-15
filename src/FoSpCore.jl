module FoSpCore
using LinearAlgebra
using IterTools
using Interpolations
using SparseArrayKit
using TensorOperations
using VectorInterface


include("./FockStates.jl")
include("./MutableFockStates.jl")
include("./FockOps.jl")
include("./NormalOrder.jl")
include("./LatticeGeo.jl")
include("./OperatorConstruction.jl")
include("./utils.jl")
include("./MatrixFree.jl")

#####################################################################################################
#####################################################################################################


export AbstractFockSpace, U1FockSpace, UnrestrictedFockSpace,
       AbstractFockState, FockState, MultipleFockState, ZeroFockState
export fock_state, copy, cleanup_FS, remove_zeros, checkU1
export a_j, ad_j
export norm2FS
export create_MFS, dot
export all_states_U1, all_states_U1_O, bounded_compositions, basisFS

#####################################################################################################
#####################################################################################################

export MutableFockState, MultipleMutableFockState, MutableFockVector, to_fock_state, reset!, reset2!, norm2FS, cleanup_FS, mul_Mutable!
export a_j!, ad_j!, apply!, key_from_occup

#####################################################################################################
#####################################################################################################


export AbstractFockOperator, FockOperator, MultipleFockOperator, ZeroFockOperator, identity_fockoperator
export cleanup_FO, dagger_FO
export apply!, apply
export rand_superpos


#####################################################################################################
#####################################################################################################

export AbstractFockString, SameSiteString, MultiSiteString
export commute_first!, normal_order!, _normal_order!, normal_order, commutator

#####################################################################################################
#####################################################################################################

export vectorise_lattice, lattice_vectorisation_map, Lattice_NN, vector_to_lattice, Lattice, AbstractLattice
export Lattice_NN_h, vectorise_NN

#####################################################################################################
#####################################################################################################

export delta, nbody_geometry, fill_nbody_tensor, make_index
export periodic_neighbour, neighbour, helical_neighbour, helical_periodic

#####################################################################################################
#####################################################################################################

export ManyBodyTensor, ManyBodyTensor_init, nbody_Op, extract_nbody_tensors, construct_Multiple_Operator, vectorize_tensor, split_tuple, devectorize_tensor
export transform

#####################################################################################################
#####################################################################################################

export MatrixFreeOperator, transition_representation

end
