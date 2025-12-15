using BenchmarkTools
using Plots
using Statistics
using FoSpCore

# Parameters
N_list = 1:3
geometry_list = [(10,10), (15,10), (20,10)]  # can extend
n_geom = length(geometry_list)
n_N = length(N_list)

# Prepare arrays to store runtimes
time_U1_O = zeros(n_geom, n_N)
time_U1 = zeros(n_geom, n_N)
system_sizes = [Lx*Ly for (Lx,Ly) in geometry_list]

# Run benchmark
for (i_geom, (Lx,Ly)) in enumerate(geometry_list)
    lattice = Lattice((Lx,Ly); periodic=(true,false), helical=(false,true))
    for (i_N, N) in enumerate(N_list)
        V = U1FockSpace((Lx,Ly), N, N)
        
        time_U1_O[i_geom, i_N] = @belapsed all_states_U1_O($V)
        time_U1[i_geom, i_N] = @belapsed all_states_U1()
    end
end

# Find common color scale (log scale recommended)
vmin = min(minimum(time_U1_O), minimum(time_U1))
vmax = max(maximum(time_U1_O), maximum(time_U1))

# Plot heatmaps
plot(
    heatmap(N_list, system_sizes, time_U1_O; color=:viridis, clim=(vmin,vmax),
            xlabel="N", ylabel="Lx*Ly", title="all_states_U1_O(V)", yflip=true),
    heatmap(N_list, system_sizes, time_U1; color=:viridis, clim=(vmin,vmax),
            xlabel="N", ylabel="Lx*Ly", title="all_states_U1()", yflip=true),
    layout = (1,2),
    size=(900,400)
)
