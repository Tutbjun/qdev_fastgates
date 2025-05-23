import numpy as np
from numpy import linalg as LA
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

import os
import sys
import math as math
import matplotlib

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.widgets import Slider, Button

matplotlib.use("TkAgg")

import cmcrameri.cm as cmc  # scientific colormaps, perceptually uniform for all.
from cmcrameri import show_cmaps  # demonstrating all the colormaps

from matplotlib import rcParams, cycler  # runtime mpl layout modifications
from matplotlib import cm

plt.style.use(
    r"C:\Users\ncl149\OneDrive - University of Copenhagen\Device Simulations\style.mplstyle"
)


# Use sparce matrices

save_plots = True
show_plots = True
adjust = True
colors = plt.cm.Set3(np.linspace(0.3, 1.1, 8))


periodicity = 2 * np.pi
number_levels = 4


# Define fluxonium potential
def potential(E_J, E_L, phi, phi_ext):
    return -E_J * np.cos(phi) + (1 / 2) * E_L * ((phi + phi_ext) ** 2) + 1


# define the fluxonium potential with a coordinate transformation
def potential_transformation(E_J, E_L, phi, phi_ext):
    return (-E_J * np.cos(phi - phi_ext) + 1 / 2 * E_L * ((phi) ** 2)) - 1.9

    # the coordinate transformation simply shiftes the potential to the side with the value of phi_ext.


def hamiltonian(E_J, E_L, E_C, phi_ext, N, phi):
    delta = phi[3] - phi[2]
    # e = 1.60217663*10**(-19) # electron charge in coulombs
    e = 1

    # # make our matrix phi
    Phi = np.zeros((N, N))
    for i in range(N):
        Phi[i][i] = phi[i]

    # # q^2 approximated:
    a = np.ones((1, N - 1))[0]
    b = np.ones((1, N))[0]
    q_2 = np.dot(
        (np.diag(-2 * b, 0) + np.diag(a, -1) + np.diag(a, 1)), (-(1)) / (delta**2)
    )
    # n_2 = np.square(1/(2*e))*q_2

    # Conductor term: kinetic energy
    C = np.dot(4 * E_C, q_2)

    # JJ term: should be a positive diagonal matrix.
    JJ = np.zeros((N, N))
    for i in range(N):
        JJ[i][i] = E_J * np.cos(Phi[i][i] - phi_ext)

    # # Inductor term: positiv diagonal matrix
    inductor = np.zeros((N, N))
    for i in range(N):
        inductor[i][i] = 1 / 2 * E_L * (Phi[i][i]) ** 2

    # Define the Hamiltonian
    Hamiltonian = C - JJ + inductor

    eig_vals, eig_vec = sp.linalg.eigh(Hamiltonian)
    # print(f"eigenvalues",eig_vals)
    return eig_vals, eig_vec


# Define initial parameters - which is also the ones i am going to find eigenvalues from!
init_phi_ext = np.pi
init_E_J = 4.0
init_E_C = 1.0
init_E_L = 1.0
N = 100 + 1
phi = np.linspace(-periodicity, periodicity, N)

eig_vals, eig_vec = hamiltonian(init_E_J, init_E_L, init_E_C, init_phi_ext, N, phi)
print(eig_vals[1] - eig_vals[0])
print(eig_vals[2] - eig_vals[1])

line = {}
for x in range(1, 5):
    line["line{0}".format(x)] = 4 * eig_vec.T[x] + eig_vals[x]

# line, = plt.plot(phi, potential_transformation(init_E_J, init_E_L,phi, init_phi_ext))


if show_plots == True:
    # Define the range of the variables

    # create figure
    fig, ax = plt.subplots()
    (line,) = ax.plot(
        phi, potential_transformation(init_E_J, init_E_L, phi, init_phi_ext)
    )
    # line2, = ax.plot(phi, potential(init_E_J, init_E_L,phi, init_phi_ext), lw=2)
    # line3, = ax.plot(phi,potential(init_E_J, init_E_L,phi, init_phi_ext),'ko')

    eig_vals, eig_vec = hamiltonian(init_E_J, init_E_L, init_E_C, init_phi_ext, N, phi)
    eig_vals = eig_vals - 2.1
    wavefunctions = {}
    lines = {}
    for x in range(0, 5):
        (wavefunctions["line{0}".format(x)],) = ax.plot(
            phi, (5 * eig_vec.T[x] + eig_vals[x]), label=f"\u03a8_{x}"
        )
        lines["line{0}".format(x)] = ax.axhline(
            y=eig_vals[x],
            color=wavefunctions["line{0}".format(x)].get_color(),
            linestyle="--",
            lw=1,
        )
        # add aotation to the energy levels |n>
        label_text = r"$|{0}\rangle$".format(x)
        ax.text(
            periodicity + 0.2,
            eig_vals[x],
            label_text,
            color=wavefunctions["line{0}".format(x)].get_color(),
        )

    ax.text(0.0, 17.0, r"$\phi_{ext} = \pi$", ha="center", va="center")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"Energy/$h$ (GHz) ")
    ax.set_ylim([-2, 18])
    ax.set_xlim([-periodicity, periodicity])
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda val, pos: "{:.0g}$\pi$".format(val / np.pi) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(plt.MultipleLocator(base=np.pi))
    # ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.suptitle('Energy levels of Fluxonium', fontsize=16)

    if adjust:
        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0.1, bottom=0.3)

        # Make horizontal sliders to control the phi_ext.
        ax_phi_ext = fig.add_axes([0.1, 0.2, 0.7, 0.04])
        phi_ext_slider = Slider(
            ax=ax_phi_ext,
            label="\u03c6_ext",
            valmin=-2 * np.pi,
            valmax=2 * np.pi,
            valinit=init_phi_ext,
        )

        # Make horizontal sliders to control the Josephson energy.
        ax_E_J = fig.add_axes([0.1, 0.15, 0.7, 0.04])
        E_J_slider = Slider(
            ax=ax_E_J,
            label="$E_J$/h (GHz)",
            valmin=0.0,
            valmax=10,
            valinit=init_E_J,
        )

        # Make horizontal sliders to control the induction energy.
        ax_E_L = fig.add_axes([0.1, 0.1, 0.7, 0.04])
        E_L_slider = Slider(
            ax=ax_E_L,
            label="$E_L$/h (GHz)",
            valmin=0.0,
            valmax=10,
            valinit=init_E_L,
        )

        # Make horizontal sliders to control the capacitor energy.
        ax_E_C = fig.add_axes([0.1, 0.05, 0.7, 0.04])
        E_C_slider = Slider(
            ax=ax_E_C,
            label="$E_C$/h (GHz)",
            valmin=0.0,
            valmax=20,
            valinit=init_E_C,
        )

        # The function to be called anytime a slider's value changes
        def update(val):
            line.set_ydata(
                potential_transformation(
                    E_J_slider.val, E_L_slider.val, phi, phi_ext_slider.val
                )
            )
            eig_vals, eig_vec = hamiltonian(
                E_J_slider.val,
                E_L_slider.val,
                E_C_slider.val,
                phi_ext_slider.val,
                N,
                phi,
            )
            for x in range(0, 5):
                wavefunctions["line{0}".format(x)].set_ydata(
                    10 * eig_vec.T[x] + eig_vals[x]
                )
                lines["line{0}".format(x)].set_ydata(eig_vals[x])
            # line2.set_ydata(potential_transformation(E_J_slider.val, E_L_slider.val,phi, phi_ext_slider.val))
            fig.canvas.draw_idle()

        phi_ext_slider.on_changed(update)
        E_J_slider.on_changed(update)
        E_L_slider.on_changed(update)
        E_C_slider.on_changed(update)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = fig.add_axes([0.8, 0.0, 0.1, 0.04])
        button = Button(resetax, "Reset", hovercolor="0.975")

        def reset(event):
            phi_ext_slider.reset()
            E_J_slider.reset()
            E_L_slider.reset()
            E_C_slider.reset()

        button.on_clicked(reset)

plt.tight_layout()
plt.show()

# if save_plots == True:
#     path = r"C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Report\Images"
#     name_file = "energy_levels_fluxonium.pdf"
#     fig.savefig(os.path.join(path, name_file), bbox_inches="tight")
