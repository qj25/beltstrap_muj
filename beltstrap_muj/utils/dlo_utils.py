import numpy as np

def quat_to_mat(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

def displaced_cable_positions(positions, quaternions, dist=None):
    """
    Compute displaced cable positions along local y-axis.

    positions: (N, 3) array of world positions.
    quaternions: (N, 4) array of [w, x, y, z] orientations.
    dist: displacement along local +y axis.

    Returns
    -------
    displaced_positions: (N, 3) array
    """
    if dist is None:
        dist = np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    N = positions.shape[0]
    displaced = np.zeros_like(positions)
    local_offset = np.array([0.0, 0.0, dist])

    for i in range(N):
        R = quat_to_mat(quaternions[i])
        offset_world = R @ local_offset
        displaced[i] = positions[i] + offset_world

    return displaced

def rot_vec(vec_0, u, a):
    # for rotation of 3d vector 
    # vec_0: vector
    # u: anchor vector (rotate about this vector)
    # a: angle in radians
    u = u / np.linalg.norm(u)
    R = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i == j:
                R[i,j] = (
                    np.cos(a)
                    + ((u[i]**2) * (1 - np.cos(a)))
                )
            else:
                ss = 1.
                if i < j:
                    ss *= -1
                if ((i+1)*(j+1)) % 2 != 0:
                    ss *= -1
                R[i,j] = (
                    u[i]*u[j] * (1 - np.cos(a))
                    + ss * u[3-(i+j)] * np.sin(a)
                )
    return np.dot(R,vec_0)

def ang_btwn(v1, v2):
    # vectors point away from the point where angle is taken
    # if np.linalg.norm(v1-v2) < 1e-6:
    #     return 0.
    cos_ab = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if cos_ab > 1:
        return 0.
    ab = np.arccos(cos_ab)
    if np.isnan(ab):
        print(np.linalg.norm(v1))
        input(np.linalg.norm(v2))
    # if np.isnan(ab):
    #     print(np.dot(v1, v2)
    #     / (
    #         np.linalg.norm(v1)
    #         * np.linalg.norm(v2)
    #     ))
    #     print(np.dot(v1, v2))
    #     print(v1)
    #     print(np.linalg.norm(v1))
    #     print(v2)
    #     print(np.linalg.norm(v2))
    #     print(np.linalg.norm(v1)*np.linalg.norm(v2))
    #     print(ab)
    #     print('here')
    return ab

def ang_btwn2(v1, v2, v_anchor):
    # use sin and cos to find angle diff from -np.pi to np.pi
    # rotation angle of v1 to v2 wrt to axis v_anchor
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v_anchor /= np.linalg.norm(v_anchor)
    e_tol = 1e-3
    if np.linalg.norm(v1-v2) < e_tol:
        return 0.
    sab = 0.
    cross12 = np.cross(v1, v2)
    # if np.linalg.norm(np.cross(cross12, v_anchor)) > e_tol:
    #     print(np.linalg.norm(np.cross(cross12, v_anchor)))
    #     print(np.linalg.norm(v1-v2))
    #     print('v1v2')
    #     print(v1)
    #     print(v2)
    #     print(cross12)
    #     print(v_anchor)
    #     raise Exception("inaccurate anchor")
    n_div = 0.
    for i in range(len(v1)):
        if abs(v_anchor[i]) < e_tol:
            continue
        sab += (
            cross12[i]
            / (
                np.linalg.norm(v1)
                * np.linalg.norm(v2)
            )
            / v_anchor[i]
        )
        n_div += 1.
    sab /= n_div
    cab = (
        np.dot(v1, v2)
        / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )
    ab = np.arctan2(sab, cab)
    return ab

def ang_btwn3(v1, v2, v_anchor):
    theta_diff = np.arccos(v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    if np.cross(v1,v2).dot(v_anchor) < 0:
        theta_diff *= -1.
    return theta_diff

def create_skewsym(v1):
    return np.array(
        [
            [0., -v1[2], v1[1]],
            [v1[2], 0., -v1[0]],
            [-v1[1], v1[0], 0.]
        ]
    )

def create_distmatrix(adjacent_distvec):
    # converts a nx3 matrix of distvecs between
    # adjacent nodes into a distvec matrix between all nodes

    # distmat is such that distmat[a,b] is the dist vec
    # from node a to b (b-a)
    n_nodes = len(adjacent_distvec)+1
    distmat = np.zeros((n_nodes,n_nodes,3))
    dist_arr = adjacent_distvec.copy()
    indcs = np.arange(len(dist_arr))
    distmat[:-1,1:][indcs,indcs,:] = dist_arr.copy()
    """
    [0,A,a,x]
    [0,0,B,b]
    [0,0,0,C]
    [0,0,0,0]

    a = A + B
    b = B + C
    x = a + C = A + B + C
    """
    for i in range(2,n_nodes):
        distmat[:i-1,i] = distmat[:i-1,i-1] + distmat[i-1,i]
    distmat = distmat - np.transpose(distmat, (1, 0, 2))
    return distmat

def get_torqvec(forcevec, distmat, ids_i):
    """
    distmat is such that distmat[a,b] is the dist vec
    from node a to b (b-a)

    MuJoCo model of DER is such that B_N is the parent of B_(N+1)
    Since DER computation is from B_N to B_first(B_0),
    a reversed order from the MuJoCo model,

            F_2 (Upwards)
    B_3 --> B_2 --> B_1 --> B_0

    Case A (moment induced at B_(N+1) by force at B_N):
    For a force of F_2 upwards at B_2, a CCW moment is induced at B_3
    M3 = [F_2 cross (B_2 - B_3)]
    this moment is applied on B_3 from B_2. 
    To apply that torque through applyFT to qfrc_passive,
    we must find the torque on B_2 from B_3, which is the negative of M3

    Case B (moment induced at B_N by force at B_(N+1)):
    For a force of F_2 upwards at B_2, a CW moment is induced at B_1
    M1 = [F_2 cross (B_2 - B_1)]
    this moment is applied on B_1 from B_2. 
    To apply that torque through applyFT to qfrc_passive,
    we must find the torque on B_1 from B_2, which is exactly M1

    Thereforce to simplify calculations,
    we will make the upper triangular matrix 
    (distances from a to b (b-a) such that b > a) negative.
        Note: This has the effect of making the whole distance matrix
        give only one way distances, i.e.
        (distmat[a,b] = distmat[b,a] = distvec of b to a (a-b) such that a < b)
        a < b because index of DER is reversed 
            -- derbodyid_a < derbodyid_b is vecbodyid_a > vecbodyid_a
    """
    triu_distmat = (
        distmat 
        * np.triu(
            np.ones((distmat.shape[0], distmat.shape[1]), dtype=bool)
        )[:,:,np.newaxis]
    )
    distmat -= 2. * triu_distmat

    n_nodes = len(distmat)
    torq_vec = np.zeros((n_nodes,3))
    for i in ids_i:
    # for i in range(len(forcevec)):
        # print(i)
        # print(distmat[:,i])
        # print(forcevec[i])
        # print(np.cross(
            # distmat[:,i],
            # forcevec[i]
        # ))
        # input()
        torqvec_indiv = np.cross(
            distmat[:,i],
            forcevec[i]
        )
        # for i2 in range(len(torqvec_indiv)):
        #     if i2 < i:
        #         torqvec_indiv[i2] = np.zeros(3)
        torq_vec += torqvec_indiv
    # for i in range(len(torqvec_indiv)):
    #     print(i)
    #     print(torq_vec[i])

    return torq_vec

def force2torq(forcevec, adjacent_distvec, ids_i):
    distmat = create_distmatrix(adjacent_distvec)
    return get_torqvec(forcevec, distmat, ids_i)

def interpolate_chain_positions(positions, new_len):
    """
    Interpolates a discretized chain of 3D positions to a new number of nodes.

    Args:
        positions (np.ndarray): Array of shape (N, 3) with original node positions.
        new_len (int): Desired number of nodes in the output.

    Returns:
        np.ndarray: Array of shape (new_len, 3) with interpolated positions.
    """
    positions = np.asarray(positions)
    N = positions.shape[0]
    if new_len == N:
        return positions.copy()
    # Compute cumulative arc-length
    diffs = np.diff(positions, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = arc_lengths[-1]
    # New arc-lengths to sample at
    new_arc_lengths = np.linspace(0, total_length, new_len)
    # Interpolate for each coordinate
    new_positions = np.empty((new_len, 3))
    for i in range(3):
        new_positions[:, i] = np.interp(new_arc_lengths, arc_lengths, positions[:, i])
    return new_positions

# def resample_chain_equal_segments(positions, new_len):
#     """
#     Resample a discretized chain of 3D positions so that all segments have equal length.

#     Args:
#         positions (np.ndarray): Array of shape (N, 3) with original node positions.
#         new_len (int): Desired number of nodes in the output.

#     Returns:
#         np.ndarray: Array of shape (new_len, 3) with resampled positions.
#     """
#     positions = np.asarray(positions)
#     N = positions.shape[0]
#     if new_len == N:
#         return positions.copy()
#     # Compute cumulative arc-length
#     diffs = np.diff(positions, axis=0)
#     seg_lengths = np.linalg.norm(diffs, axis=1)
#     arc_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
#     total_length = arc_lengths[-1]
#     # New arc-lengths to sample at, equally spaced
#     new_arc_lengths = np.linspace(0, total_length, new_len)
#     # Interpolate for each coordinate
#     new_positions = np.empty((new_len, 3))
#     for i in range(3):
#         new_positions[:, i] = np.interp(new_arc_lengths, arc_lengths, positions[:, i])
#     return new_positions

def resample_chain_exact_equal_segments(positions, new_len):
    """
    Resample a chain of 3D positions so that all segments have exactly equal length.

    Args:
        positions (np.ndarray): (N, 3) original node positions.
        new_len (int): Number of nodes in the output.

    Returns:
        np.ndarray: (new_len, 3) resampled positions with exactly equal segment lengths.
    """
    positions = np.asarray(positions)
    N = positions.shape[0]
    if new_len == N:
        return positions.copy()
    # Compute cumulative arc-length
    diffs = np.diff(positions, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = arc_lengths[-1]
    segment_length = total_length / (new_len - 1)
    new_positions = [positions[0]]
    seg_idx = 0
    for i in range(1, new_len):
        target_arc = i * segment_length
        # Move seg_idx to the segment containing target_arc
        while seg_idx < len(arc_lengths) - 2 and arc_lengths[seg_idx+1] < target_arc:
            seg_idx += 1
        # Interpolate between positions[seg_idx] and positions[seg_idx+1]
        t = (target_arc - arc_lengths[seg_idx]) / (arc_lengths[seg_idx+1] - arc_lengths[seg_idx])
        new_pt = (1 - t) * positions[seg_idx] + t * positions[seg_idx+1]
        new_positions.append(new_pt)
    return np.array(new_positions)

if __name__ == "__main__":
    v1 = np.array([1.5, 1.5, 0.])
    v2 = np.array([-1., 0., 0.])
    va = np.array([0., 1., 0.])
    va = np.cross(v1,v2)
    va *= 1.

    print(ang_btwn2(v1,v2,va)/np.pi*180)
    print(ang_btwn3(v1,v2,va)/np.pi*180)