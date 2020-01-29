import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import RK45
import matplotlib.animation as animation
import time

#___________________________________________ CONTROLS _____________________________________________#
intvar = 1
gravity_on = 1
twoplanets = 1

#_______________________________________ INITIAL CONDITIONS _______________________________________#
# Legend: x, y, z, vx, vy, vz, m, rho, p, energy

# Define initial state vector
initialcond = np.loadtxt('Planet300.dat')
particles = len(initialcond)

# Define parameters
kappa = 2 # Used in the NNPS algorithm
nu = 1.4 # Scaling factor for the smoothing factor
gamma_sound = 1.4 # For the sound speed
h_constant = 1e7
G = 6.67408e-11

# Separate into variables to make it easier
x = initialcond[:,0]
y = initialcond[:,1]
z = initialcond[:,2]
vx = initialcond[:,3]
vy = initialcond[:,4]
vz = initialcond[:,5]
rho = initialcond[:,7]
pressure = initialcond[:,8]

if twoplanets == 0:
    e_0 = initialcond[:,8]/((gamma_sound - 1)*initialcond[:,7])   
    # Initial state vector 
    S = np.zeros((particles, len(initialcond[0])))
    S[:,0] = x
    S[:,1] = y 
    S[:,2] = z
    S[:,3] = vx
    S[:,4] = vy
    S[:,5] = vz
    S[:,6] = rho
    S[:,7] = pressure
    S[:,8] = e_0
    m = initialcond[:,6]
        
if twoplanets == 1:
    e_0 = initialcond[:,8]/((gamma_sound - 1)*initialcond[:,7])       
    # Initial state vector 
    S = np.zeros((2*particles, len(initialcond[0])))
    S[:,0] = np.reshape(np.array((x - 5e7, x + 5e7)), (particles*2)) # x for the two planets
    S[:,1] = np.reshape(np.array((y - 5e7, y + 5e7)), (particles*2))# y 
    S[:,2] = np.reshape(np.array((z, z)), (particles*2)) # z
    S[:,3] = np.reshape(np.array((vx + 1000, vx - 10000)), (particles*2)) # vx
    S[:,4] = np.reshape(np.array((vy + 1000, vy - 1000)), (particles*2)) # vy
    S[:,5] = np.reshape(np.array((vz, vz)), (particles*2)) # vz
    mass = np.reshape(np.array((initialcond[:,6], initialcond[:,6]*100)), (particles*2))   # mass
    S[:,6] = np.reshape(np.array((rho,rho)), (particles*2)) # rho
    S[:,7] = np.reshape(np.array((pressure,pressure)), (particles*2)) # pressure
    S[:,8] = np.reshape(np.array((e_0,e_0)), (particles*2)) # energy

fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot((111), projection='3d')
ax.scatter(S[:,0], S[:,1], S[:,2])#, c='cs', marker='m')
ax.set_xlim(-2e8, 2e8)
ax.set_xlabel('$ x \, [m] $')
ax.set_ylim(-2e8, 2e8)
ax.set_ylabel('$ y \, [m] $')
ax.set_zlim(-2e8, 2e8)
ax.set_zlabel('$ z \, [m] $')

# Reshape initial state vector
N = len(S) # Numer of particles
nparams = len(S[0])
S = S.reshape(N*nparams) 


#________________________________________ FUNCTIONS _______________________________________________#
def h_len(mass, density):
    """ Calculates the smoothing length for all the particles for a given
    state vector. """
    return np.zeros(N) + h_constant # nu*(mass/density)**((1/3))

hlen = np.full(N, h_constant)

def smoothingW(r, hmean):
    """ Utilizes the relative distances of a pair to calculate the 
    smoothing function. The input relative distanc has already been calculated
    with the smoothing factor h."""
    
    ad = (3/(2*np.pi*hmean**3)) # Alpha-d factor
    R = r/hmean
    smoothW = np.zeros(len(R))
    
    # Define masks
    mask_01 = (R >= 0) & (R < 1) # First condition
    mask_02 = (R >= 1) & (R < 2) # Second condition
  
    # Calculate all values for the smoothing function given the conditions
    smoothW[mask_01] = ad[mask_01]*(2/3 - (R[mask_01])**2 + 0.5*(R[mask_01])**3)
    smoothW[mask_02] = ad[mask_02]*((2-(R[mask_02]))**3)/6   
                  
    return smoothW

def smoothingdW(r, dX, hmean):
    """ Utilizes the relative distances of a pair in all three coordinates to calculate the 
    derivative of the smoothing function. """
    
    ad = (3/(2*np.pi*hmean**3)) # Alpha-d factor
    R = r/hmean
    smoothdW = np.zeros((3, len(R))) 
    
    # Define masks
    mask_01 = (R >= 0) & (R < 1) # First condition
    mask_02 = (R >= 1) & (R < 2) # Second condition
      
    # Stack individual masked vectors into arrays
    dX_1 = dX[:,mask_01]
    dX_2 = dX[:,mask_02]
    
    # Calculate all values for the derivatives of the smoothing function given the conditions
    constant1 = ad[mask_01]*(-2 + 1.5*(R[mask_01]))/(hmean[mask_01]**2)    
    smoothdW[:,mask_01] = constant1*(dX_1)
    
    constant2 = -ad[mask_02]*(0.5*((2-(R[mask_02]))**2))/(hmean[mask_02]*r[mask_02])    
    smoothdW[:,mask_02] = constant2*(dX_2)

    
    return smoothdW # Output has three components, one for each direction of the dx vector    

def artvisc(r, dX, dV, rhomean, hmean, cmean, dot):
    # Define parameters that go into the artificial viscosity

    a = 1
    b = 1
    
    # Relative quantities
    phi = 0.1*hmean
    
    # Output of the viscosity is also one for each coordinate
    theta = (hmean*dot)/(abs(r)**2 + phi**2)   
    artvisc = (-a*cmean*theta + b*theta*theta)/(rhomean)

    return artvisc

def potential(r, dX, hmean):
    # Calculates the gravitational potential for a given pair
    # Calculate R, same as in the smoothing functions
    R = r/hmean
    
    # Create masks for the different conditions
    mask_01 = (R >= 0) & (R < 1) # First condition
    mask_02 = (R >= 1) & (R < 2) # Second condition
    mask_03 = (R >= 2) # Third condition
    
    potentials = np.zeros((3, (len(dX[0])))) # One potential value for each pair (according to second eq.)
    norm = dX/r
    
    potentials[:,mask_01] = (1/((hmean[mask_01] + 0.1)**2))*((4*R[mask_01])/3 
             - (6*(R[mask_01]**3)/5) + 0.5*R[mask_01]**4)*norm[:,mask_01]
    
    potentials[:,mask_02] = (1/((hmean[mask_02])**2))*((8/3)*R[mask_02] - 3*R[mask_02]**2 
             + (6/5)*R[mask_02]**3 - (1/6)*R[mask_02]**4 - 1/(15*(R[mask_02])**2))*norm[:,mask_02]
    
    potentials[:,mask_03] = 1/((r[mask_03])**2)**norm[:,mask_03]
    
    return potentials

def gravity(mass, dX, potentials):
    """ Gives the velocity change caused by the gravitational interaction between the pairs 
    particles. """
    
    norm = np.linalg.norm(dX, axis=0)
    gravint = -0.5*G*mass*(2*potentials)*(dX/norm)
    
    return gravint
    
def velocity(smoothingdW, mass, pressure1, pressure2, density1, density2, artvisc):
    """ Computes de derivative of the velocity. """     
    # Remember that smoothing dW and the artificial viscosity are vector quantities
    # Compute the sum of the ratios of the density and pressure
    
    ratio = (pressure1/(density1*density1)) + (pressure2/(density2*density2))
    d_V = -mass*(ratio + artvisc)*smoothingdW
    
    return d_V

def energy(mass, density1, density2, pressure1, pressure2, dV, smoothingdW, artvisc):
    
    return 0.5*mass*((pressure1/(density1*density1)) + (pressure2/(density2*density2)) 
                     + artvisc)*np.sum(dV*smoothingdW, axis=0) 
    
#________________________________________ 3D NNPS ALGORITHM _______________________________________#
def NNPScalc(S):
    """ Nearest neightbour pair search algorithm - vectorized version. Structure follows the 
    for loop version. Calculates all the quantities involving pairs. """
            
#    S[:,0] = x
#    S[:,1] = y
#    S[:,2] = z
#    S[:,3] = vx
#    S[:,4] = vy
#    S[:,5] = vz
#    S[:,6] = density rho
#    S[:,7] = pressure
#    S[:,8] = energy
    
    # Reshape input vector
    S = S.reshape(N, nparams) 

#    # Calculate all possible combinations of the smoothing length as a matrix.
#    h_length = h_len(S[:,6], S[:,7])
#    hmean = (h_length.reshape(N, 1) +  h_length)*0.5
    
    hmean = np.full([N,N], h_constant)

    trii, trij = np.triu_indices(len(hmean), k=1)
    
    # Positions: x, y, z (the u stands for unsmaked)
    dxx_u = (S[:,0].reshape(N, 1) - S[:,0])[trii, trij]
    dxy_u = (S[:,1].reshape(N, 1) - S[:,1])[trii, trij]
    dxz_u = (S[:,2].reshape(N, 1) - S[:,2])[trii, trij]
    
    # Velocities: vx, vy, vz
    dvx_u = (S[:,3].reshape(N, 1) - S[:,3])[trii, trij]
    dvy_u = (S[:,4].reshape(N, 1) - S[:,4])[trii, trij]
    dvz_u = (S[:,5].reshape(N, 1) - S[:,5])[trii, trij]
     
    # Find pairs and obtain indexes
    dX_u = np.array((dxx_u, dxy_u, dxz_u))
    dV_u = np.array((dvx_u, dvy_u, dvz_u))
    r_u = np.linalg.norm(dX_u, axis = 0)
    
    # Create nearest neighbor mask
    maskNNPS_total = (r_u <= kappa*hmean[trii, trij]) & (r_u > 0) # Vector mask

    pi = trii[maskNNPS_total]
    pj = trij[maskNNPS_total]
    
    if gravity_on == 1:
        
        # Calculate the longrange for ALL the possible pairs taking into account the antisymmetry
        # of the lower triangle
        longrange_i = -G*mass[trij]*(dX_u/(r_u*r_u*r_u))
        longrange_j =  G*mass[trii]*(dX_u/(r_u*r_u*r_u)) 
        
        # Set the values of the nearest neighbors to 0 
        longrange_i[:,maskNNPS_total] = 0
        longrange_j[:,maskNNPS_total] = 0
        
        # Sum all the interactions for each particle
        grav_long = np.zeros((3, N))
        np.add.at(grav_long.T, trii, longrange_i.T)
        np.add.at(grav_long.T, trij, longrange_j.T)
    
    if gravity_on == 0:
        longrange_i, longrange_j = np.zeros((3, len(trii))) , np.zeros((3, len(trii)))
        
    # Mask the difference vectors, the smoothing lenghts and the norms for the given pairs
    dX = dX_u[:, maskNNPS_total]
    dxx, dxy, dxz = dX
    
    dV = dV_u[:, maskNNPS_total]
    dvx, dvy, dvz = dV
    
    hmean =  hmean[trii, trij][maskNNPS_total]
    r = r_u[maskNNPS_total]
    
    # Calculate the smoothing function and its derivative. The input vectors are already masked
    smoothW = smoothingW(r, hmean)
    smoothdW = smoothingdW(r, dX, hmean)
    
    # Compute pair quantities 
    # Mean density of a pair
    rhomean = ((((S[:,6]).reshape(N, 1) + S[:,6])*0.5)[trii, trij])[maskNNPS_total]
    
    # Sound speed
    c = np.sqrt((gamma_sound - 1)*S[:,8]) # Sound speed    
    cmean = ((((c.reshape(len(c), 1) + c))*0.5)[trii, trij])[maskNNPS_total]
    
    # Artificial viscosity: one value per component, 3D quantity
    dot = np.sum(dV*dX, axis=0)
    maskVISC = (dot < 0) 
    viscosity = np.zeros(len(pi))
    viscosity[maskVISC] = artvisc(r[maskVISC], dX[:,maskVISC], dV[:,maskVISC],
             rhomean[maskVISC], hmean[maskVISC], cmean[maskVISC], dot[maskVISC])
        
    """ Import the indexes for the pairs, the smoothing function and the derivative of the 
    smoothing function, the viscosity values, the difference vectors for position and velocity, 
    the mean smoothing length, the absolute distances between the particles and the gravity. """
    
    return pi, pj, smoothW, smoothdW, viscosity, dX, dV, hmean, r, trii, trij, grav_long

#______________________________________ FUNCTION TO INTEGRATE _____________________________________#

def integrate(t, S):
    """ Function to integrate. """

#    S[:,0] = x
#    S[:,1] = y
#    S[:,2] = z
#    S[:,3] = vx
#    S[:,4] = vy
#    S[:,5] = vz
#    S[:,6] = density rho
#    S[:,7] = pressure
#    S[:,8] = energy
    
    # Import data from the NNPS function
    pi_s, pj_s, smoothW, smoothdW, viscosity, dX, dV, hmean, r, trii, trij, grav_long = NNPScalc(S)
   
    S = S.reshape(N, nparams)
    
    # Start derivative calculations   
    dS = np.zeros(np.shape(S)) # Empty array to store the derivatives
    
    # Calculate density with the summation density
    S[:,6] = mass/((np.pi*hlen*hlen*hlen)) # Density self effect (for every particle)
    
    density_i = mass[pj_s]*smoothW
    density_j = mass[pi_s]*smoothW
    
    np.add.at(S[:,6], pi_s, density_i)
    np.add.at(S[:,6], pj_s, density_j)

    # Update the pressures 
    S[:,7] = (gamma_sound - 1)*(S[:,6]*S[:,8]) # Updates pressure. Depends only on particle
      
    # Calculate the velocities to add them later to the vectors
    velocity_i = velocity(smoothdW, mass[pj_s], S[:,7][pi_s], S[:,7][pj_s], S[:,6][pi_s], S[:,6][pj_s], viscosity)
    velocity_j = -velocity(smoothdW, mass[pi_s], S[:,7][pj_s], S[:,7][pi_s], S[:,6][pj_s], S[:,6][pi_s], viscosity)
    
    if gravity_on == 1:
        # Calculate gravitational potential for the neighbouring particles
        potentials = potential(r, dX, hmean)
        
        # Calculate gravity between the pairs of particles and combine it with the velocity changes 
        dv_i = -G*mass[pj_s]*potentials + velocity_i
        dv_j =  G*mass[pi_s]*potentials + velocity_j
        
        # Create velocity change for all the particles to loop over later
        np.add.at(grav_long.T, pi_s, dv_i.T)
        np.add.at(grav_long.T, pj_s, dv_j.T)
        
    # Calculate the velocities to add them later to the vectors
    energy_i = energy(mass[pj_s], S[:,6][pi_s], S[:,6][pj_s], S[:,7][pi_s], S[:,7][pj_s], dV, smoothdW, viscosity)
    energy_j = energy(mass[pi_s], S[:,6][pj_s], S[:,6][pi_s], S[:,7][pj_s], S[:,7][pi_s], dV, smoothdW, viscosity)
    
    # After the calculations, assign the values to the respective arrays
    # Velocity change   
    dS[:,3] = grav_long[0]
    dS[:,4] = grav_long[1]
    dS[:,5] = grav_long[2]

    # Energy change
    np.add.at(dS[:,8], pi_s, energy_i)
    np.add.at(dS[:,8], pj_s, energy_j)
    
    # Derivative of the position are the input velocities
    dS[:,0] = S[:,3] 
    dS[:,1] = S[:,4]
    dS[:,2] = S[:,5]
    
    # Set derivatives of density and pressure to 0 
    dS[:,6] = 0
    dS[:,7] = 0
    
    return dS.reshape(N*nparams)


#_________________________________________ INTEGRATION ____________________________________________#

start_time = time.time()

if intvar == 1:
    steps = 10000
    S_int = RK45(integrate, 0, S, 1000*3600, max_step = 25.0)#, rtol=0.001, atol=1e-6)
    S_i = np.zeros([steps, N, nparams])
    
    # Loop for the integration
    for i in range(steps):
        S_i[i] = np.array(S_int.y).reshape(N, nparams) # Select current state, reshape and store
        S_int.step()        
        print(i, S_int.step_size)

    # Thanks Lukas, you fam 
    fig = plt.figure()
    ax = fig.add_subplot((111), projection='3d')
    for i in range(1500):
        if i % 15:
            ax.scatter(S_i[i,:,0], S_i[i,:,1], S_i[i,:,2], c=S_i[i,:,6], s=5, cmap='hot')
            ax.set_xlim(-2e8, 2e8)
            ax.set_xlabel('$ x \, [m] $')
            ax.set_ylim(-2e8, 2e8)
            ax.set_ylabel('$ y \, [m] $')
            ax.set_zlim(-2e8, 2e8)
            ax.set_zlabel('$ z \, [m] $')
#           ax.scatter(S_i[i,:,0], S_i[i,:,1])
            ax.set_title('\n Step %4.f  \n' % i )
            plt.tight_layout()
            plt.pause(0.1)
            ax.clear()

print("--- %s seconds ---" % (time.time() - start_time))
np.save('300planet-diffmass-headon.npy', S_i)

#_________________________________________ HOLLYWOOD ______________________________________________#
# Plots for the report 

# Import saved data
S_i_1200 = np.load('1200planet.npy')
S_i_300 = np.load('300planet-equalmass-headon.npy')

# Select data to plot
S_plot = S_i_1200

i = 1150
time = i*25 
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot((111), projection='3d')
p = ax.scatter(S_plot[i,:,0], S_plot[i,:,1], S_plot[i,:,2], c=S_plot[i,:,6], s=5, cmap='hot')
cbar = plt.colorbar(p, ax=ax, label='Density [kg/m$^3$]')
ax.set_xlim(-1.5e8, 1.5e8)
ax.set_xlabel('$ x \, [m] $')
ax.set_ylim(-1.5e8, 1.5e8)
ax.set_ylabel('$ y \, [m] $')
ax.set_zlim(-1.5e8, 1.5e8)
ax.set_zlabel('$ z \, [m] $')
#ax.set_title('Time %4.f\ns' % time)
plt.tight_layout()
plt.savefig('1200edgeon-1150.png', dpi=300,  bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot((111), projection='3d')
for i in range(1500):
    ax.scatter(S_plot[i,:,0], S_plot[i,:,1], S_plot[i,:,2], c=S_plot[i,:,6], s=5, cmap='hot')
    ax.set_xlim(-2e8, 2e8)
    ax.set_xlabel('$ x \, [m] $')
    ax.set_ylim(-2e8, 2e8)
    ax.set_ylabel('$ y \, [m] $')
    ax.set_zlim(-2e8, 2e8)
    ax.set_zlabel('$ z \, [m] $')
#           ax.scatter(S_i[i,:,0], S_i[i,:,1])
    ax.set_title('\n Step %4.f  \n' % i )
    plt.tight_layout()
    plt.pause(0.1)
    ax.clear()