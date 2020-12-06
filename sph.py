import taichi as ti
ti.init(arch=ti.opengl)

n_particles = 2500
dt = 0.005
p_mass = 1
K = 0.01
rho_rest = 1000
gravity = ti.Vector([0,-9.8])
PI = 3.141
VISC_CONS = 0.0225
BOUND = 0.8
BOUND_DAMPING = -1
H = 1

H2 = H*H
POLY6 = 315.0/(64.0*PI*pow(H,9))
SPIKY = -3.0 * 15.0 / (PI * pow(H,6))
VISC = 45.0 / (PI * pow(H,6))

x = ti.Vector.field(2, float, n_particles) #position
v = ti.Vector.field(2, float, n_particles) #velocity
rho = ti.field(float, n_particles) #density
p = ti.field(float, n_particles) #pressure
f = ti.Vector.field(2, float, n_particles) #force

@ti.func
def Wpoly6(r2):
    return POLY6 * pow(H2-r2,3)

@ti.func
def Wspiky(r):
    return SPIKY * pow(H-r,2)

@ti.func
def Wvisc(r):
    return VISC * (H-r)

@ti.kernel
def calDensPress():
    for i in range(n_particles):
        rho[i] = 0
        for j in range(n_particles):
            rij = x[i] - x[j]
            r2 = rij.norm_sqr()
            if r2 < H2:
                rho[i] += p_mass * Wpoly6(r2)
        p[i] = K * (rho[i]-rho_rest)

@ti.kernel
def calForce():
    for i in range(n_particles):
        fPress = ti.Vector.zero(float, 2)
        fVisc = ti.Vector.zero(float, 2)
        for j in range(n_particles):
            if i == j:
                continue
            rij = x[i] - x[j]
            r = rij.norm()
            if r < H:
                fPress += -rij.normalized() * (p[i] + p[j]) / (2 * rho[j]) * Wspiky(r)
                fVisc += (v[j] - v[i]) / (rho[j]) * Wvisc(r)
        fPress *= p_mass
        fVisc *= VISC_CONS * p_mass
        fGrav = gravity * rho[i]
        f[i] = fPress + fVisc + fGrav

@ti.func
def collide(i):
    if x[i][0] < 0.2:
        v[i][0] *= BOUND_DAMPING
        x[i][0] = 0.2
    if x[i][1] < 0.2:
        v[i][1] *= BOUND_DAMPING
        x[i][1] = 0.2
    if x[i][0] > BOUND:
        v[i][0] *= BOUND_DAMPING
        x[i][0] = BOUND
    if x[i][1] > BOUND:
        v[i][1] *= BOUND_DAMPING
        x[i][1] = BOUND

@ti.kernel
def integrate():
    for i in range(n_particles):
        v[i] += dt * f[i] / rho[i]
        x[i] += dt * v[i]
        collide(i)

@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]

init()
gui = ti.GUI('SPH')
while gui.running and not gui.get_event(gui.ESCAPE):
    calDensPress()
    calForce()
    integrate()
    # avg = 0
    # for i in range(n_particles):
    #     avg += rho[i]
    # avg /= n_particles
    # print(avg)
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=4, color=0x068587)
    gui.show()
