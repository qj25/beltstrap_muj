import os 
import mujoco
import mujoco_viewer
import numpy as np
import beltstrap_muj.utils.mjc2_utils as mjc2
from beltstrap_muj.utils.xml_utils import XMLWrapper
# from beltstrap_muj.controllers.wire_plugin.WireStandalone import WireStandalone
from beltstrap_muj.utils.mjc_utils import MjSimWrapper
from beltstrap_muj.utils.mjc2_utils import init_plugins
from beltstrap_muj.utils.dlo_utils import interpolate_chain_positions, displaced_cable_positions
# from beltstrap_muj.utils.real2sim_utils import compute_wire_frames
from beltstrap_muj.assets.genrope.gen_dlo_xml import generate_dlo_xml
from beltstrap_muj.assets.genrope.gen_belt_xml import generate_belt_xml


# Settings
belt_from_circle = True
do_render = True
# update stiffness, mass, and length as needed.
# alpha_bar = 1.345
# beta_bar = 0.789
# alpha_bar = 0.001196450659614982    # Obtained from simple PI
# beta_bar = 0.001749108044378543
alpha_bar = 0.0001
beta_bar = 0.000001
mass_per_length = 0.079/2.98
thickness = 0.006
j_damp = 0.1
# j_damp = 0.02

n_nodes = 33   # adjust parllThreshold with change in n_pieces
n_steps = 50000

torq_tol = 1e-8
tolC2 = 3e-4
tolC3 = 3e-4
m_p = 1.0e-2
m_p = 1.0e-6

assets_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'beltstrap_muj/assets'
)
xml_path = os.path.join(assets_path,'belt.xml')

init_plugins()


# init stiffnesses for capsule
J1 = np.pi * (thickness/2)**4/2.
Ix = np.pi * (thickness/2)**4/4.
stiff_vals = [
    beta_bar/J1,
    alpha_bar/Ix
]

# Define shape of belt
if belt_from_circle:
    belt_initradius = 0.5
    total_length = 2 * np.pi * belt_initradius * 1.49
    mass = mass_per_length * total_length
    rgba_wire = "0.1 0.0533333 0.673333 1"
    generate_belt_xml(
        belt_initradius=belt_initradius,
        expansion_coeff=0.01,
        r_len=total_length,
        n_pieces=n_nodes-1,
        thickness=thickness,
        mass=mass,
        j_damp=j_damp,
        con_val=(1,0),
        stiff_bend=stiff_vals[1],
        stiff_twist=stiff_vals[0],
        xml_path=xml_path,
        rgba=rgba_wire
    )
else:
    wire_pos = np.array([
        [0.0,0,0],
        [0.1,0,0],
        [0.1,0.1,0],
        [0.0,0.1,0],
    ])
    wire_pos_og = wire_pos.copy()
    wire_pos = interpolate_chain_positions(wire_pos_og, n_nodes)
    
    # Generate wire xml
    # includes conversion of wire xpos to main body pose and subsequent connected quaternions
    # Compute total arc length
    segment_lengths = np.linalg.norm(np.diff(wire_pos, axis=0), axis=1)
    total_length = np.sum(segment_lengths)
    
    # Mass per unit length
    mass = mass_per_length * total_length
    # init_pos = np.array([0.0, 0.0, 0.5])
    # init_quat = np.array([1.0, 0.0, 0.0, 0.0])
    rgba_wire = "0.1 0.0533333 0.673333 1"
    generate_dlo_xml(
        n_pieces=n_nodes-1,
        thickness=thickness,
        mass=mass,
        j_damp=j_damp,
        con_val=(1,0),
        stiff_bend=stiff_vals[1],
        stiff_twist=stiff_vals[0],
        wire_pos=wire_pos,
        xml_path=xml_path,
        rgba=rgba_wire
    )
xml = XMLWrapper(xml_path)

# # Load MuJoCo model and data
# model = mujoco.MjModel.from_xml_path(xml_path)
xml_string = xml.get_xml_string()
model = mujoco.MjModel.from_xml_string(xml_string)
mujoco.mj_saveLastXML(xml_path,model)
data = mujoco.MjData(model)
# model.opt.gravity[-1] = 0.0
# model.opt.gravity[-1] = -9.81

known_body_name = "B_0"
plgn_instance = model.body_plugin[
    mjc2.obj_name2id(model, "body", known_body_name)
]
start = model.plugin_stateadr[plgn_instance]
r_len = total_length
r_pieces = n_nodes - 1
vec_bodyid = np.zeros(r_pieces, dtype=int)
for i in range(r_pieces):
    if i == 0:
        i_name = '0'
    elif i == r_pieces-1:
        i_name = 'last'
    else:
        i_name = str(i)
    vec_bodyid[i] = mjc2.obj_name2id(
        model,"body",'B_' + i_name
    )
vec_bodyid_full = np.concatenate((
    vec_bodyid, [mjc2.obj_name2id(
        model,"body",'B_last2'
    )]
))

sim = MjSimWrapper(model, data)
if do_render:
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.vopt.geomgroup[3] ^= 1
    dist=1.5
    azi=90.0
    elev=-30.0
    lookat=np.array([0.0, 0.0, 0.0])
    viewer.cam.distance = dist
    viewer.cam.azimuth = azi
    viewer.cam.elevation = elev
    viewer.cam.lookat = lookat
    viewer.render()
    viewer._paused = True
    def add_marker(position, size=0.01, rgba=(1, 0, 0, 1)):
        marker = {
            # "type": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "type": mujoco.mjtGeom.mjGEOM_SPHERE,
            "size": [size,size,size],  # [radius, half-length]
            "pos": position,
            # "mat": mat,
            "rgba": rgba,
        }
        viewer.add_marker(**marker)

sim.forward()
sim.step()
if do_render:
    viewer.render()
sim.forward()

# data.eq_active[0] = 0
# data.eq_active[1] = 0

for i in range(n_steps):
    sim.step()
    sim.forward()
    xpos = data.xpos[vec_bodyid_full]
    xquat = data.xquat[vec_bodyid_full]
    displaced_pos = displaced_cable_positions(
        xpos, xquat, dist=0.03
    )

    for pt in displaced_pos:
        add_marker(pt, size=0.01, rgba=(1, 0, 0, 1))

    if do_render:
        viewer.render()