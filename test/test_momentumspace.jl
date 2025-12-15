using Revise
using FoSpCore, Test, Plots


N = 1
geometry = (6,6)
V = U1FockSpace(geometry, N,N)
lattice = Lattice(geometry; periodic=(true,true))

Kin, Int = Bose_Hubbard_H(V, lattice, .5, 0.)

Op = MultipleFockOperator([Kin.terms...; Int.terms...])
Op = Kin
# -------------------------------
# Extract 2-body and 4-body tensors
# -------------------------------
t2 = get_tensor_2body(Op, lattice);
t4 = get_tensor_4body(Op, lattice);

@test t2[1,1,2,1] == 0.5+0im
@test t4[1,1,1,1,1,1,1,1] == 1.0+0im

# -------------------------------
# Reconstruct operators from tensors
# -------------------------------
Op2 = two_body_Op(V, lattice, t2);
Op4 = four_body_Op(V, lattice, t4);

@test Op2.terms[1].coefficient == .5+0im
@test Op4.terms[1].coefficient == 1.0+0im

# -------------------------------
# Momentum-space test
# -------------------------------
dims = (1,2)
O_m = momentum_space_Op(Op, lattice, dims);

O_m

# Example: discrete momenta
kx_vals = 1:6
ky_vals = 1:6

# Initialize plot
pl = plot(xlabel="k_x", ylabel="k_y", zlabel="ε(k_x,k_y)", 
          legend=false, title="Bose-Hubbard dispersion")

# --- scatter points ---
for kX in kx_vals, kY in ky_vals
    scatter!(pl, [kX], [kY], [real(E(kX,kY))], markersize=4, markercolor=:blue)
end

# --- surface from the analytical tensor ---
kx_grid = repeat(kx_vals, inner=length(ky_vals))
ky_grid = repeat(ky_vals, outer=length(kx_vals))
E_vals = [real(E(kX, kY)) for kX in kx_vals, kY in ky_vals]  # 5x5 matrix

surface!(pl, kx_vals, ky_vals, E_vals, alpha=0.3, color=:red)

display(pl)

