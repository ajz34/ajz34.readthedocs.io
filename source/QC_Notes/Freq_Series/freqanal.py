import numpy as np
from formchk_interface import FormchkInterface
from scipy.constants import physical_constants

# https://docs.scipy.org/doc/scipy/reference/constants.html
E_h = physical_constants["Hartree energy"][0]
a_0 = physical_constants["Bohr radius"][0]
N_A = physical_constants["Avogadro constant"][0]
c_0 = physical_constants["speed of light in vacuum"][0]
e_c = physical_constants["elementary charge"][0]
e_0 = physical_constants["electric constant"][0]
mu_0 = physical_constants["mag. constant"][0]


class FreqAnal:
    
    def __init__(self):
        self.natm = NotImplemented         # Atom number
        self.mol_weights = NotImplemented  # Molecular weight (dim: natm, unit: amu)
        self.mol_coords = NotImplemented   # Atom coordinates (dim: (natm, 3), unit: Bohr)
        self.mol_hess = NotImplemented     # Hessian matrix (dim: (natm*3, natm*3), unit: a.u.)
        self._mom_inertia = NotImplemented   # Moment of inertia (dim: (3, 3), unit: a.u.)
        self._theta = NotImplemented       # Force constant tensor
        self._proj_inv = NotImplemented    # Inverse space of translation and rotation of theta
        self._freq = NotImplemented        # Frequency (unit: cm-1)
        self._rot_vec = NotImplemented     # Moments of inertia principle axis (eigenvector, unit: None)
        self._rot_eig = NotImplemented     # Moments of inertia value (unit: a.u.)
        self._q = NotImplemented           # Unnormalized normal coordinate
        self._qnorm = NotImplemented       # Normalized normal coordinate (unit: None)
    
    def init_from_gaussian(self, fchk_path):
        fchk = FormchkInterface(fchk_path)
        self.mol_weights = fchk.key_to_value("Real atomic weights")
        self.natm = natm = self.mol_weights.size
        self.mol_coords = fchk.key_to_value("Current cartesian coordinates").reshape((natm, 3))
        self.mol_hess = fchk.hessian()
        self.mol_hess = (self.mol_hess + self.mol_hess.T) / 2
        self.mol_hess = self.mol_hess.reshape((natm, 3, natm, 3))
        return self
        
    @property
    def theta(self):
        if self._theta is NotImplemented:
            natm, mol_hess, mol_weights = self.natm, self.mol_hess, self.mol_weights
            self._theta = np.einsum("AtBs, A, B -> AtBs", mol_hess, 1 / np.sqrt(mol_weights), 1 / np.sqrt(mol_weights)).reshape(3 * natm, 3 * natm)
        return self._theta
    
    @property
    def center_coord(self):
        return (self.mol_coords * self.mol_weights[:, None]).sum(axis=0) / self.mol_weights.sum()
    
    @property
    def centered_coord(self):
        return self.mol_coords - self.center_coord
    
    def _get_rot(self):
        natm, centered_coord, mol_weights = self.natm, self.centered_coord, self.mol_weights
        rot_tmp = np.zeros((natm, 3, 3))
        rot_tmp[:, 0, 0] = centered_coord[:, 1]**2 + centered_coord[:, 2]**2
        rot_tmp[:, 1, 1] = centered_coord[:, 2]**2 + centered_coord[:, 0]**2
        rot_tmp[:, 2, 2] = centered_coord[:, 0]**2 + centered_coord[:, 1]**2
        rot_tmp[:, 0, 1] = rot_tmp[:, 1, 0] = - centered_coord[:, 0] * centered_coord[:, 1]
        rot_tmp[:, 1, 2] = rot_tmp[:, 2, 1] = - centered_coord[:, 1] * centered_coord[:, 2]
        rot_tmp[:, 2, 0] = rot_tmp[:, 0, 2] = - centered_coord[:, 2] * centered_coord[:, 0]
        rot_tmp = (rot_tmp * mol_weights[:, None, None]).sum(axis=0)
        rot_eig, rot_vec = np.linalg.eigh(rot_tmp)
        return rot_eig, rot_vec, rot_tmp
    
    @property
    def mom_inertia(self):
        if self._mom_inertia is NotImplemented:
            self._rot_eig, self._rot_vec, self._mom_inertia = self._get_rot()
        return self._mom_inertia
    
    @property
    def rot_eig(self):
        if self._rot_eig is NotImplemented:
            self._rot_eig, self._rot_vec, self._mom_inertia = self._get_rot()
        return self._rot_eig
    
    @property
    def rot_vec(self):
        if self._rot_vec is NotImplemented:
            self._rot_eig, self._rot_vec, self._mom_inertia = self._get_rot()
        return self._rot_vec
    
    @property
    def proj_scr(self):
        natm, centered_coord, rot_vec, mol_weights = self.natm, self.centered_coord, self.rot_vec, self.mol_weights
        rot_coord = np.einsum("At, ts, rw -> Asrw", centered_coord, rot_vec, rot_vec)
        proj_scr = np.zeros((natm, 3, 6))
        proj_scr[:, (0, 1, 2), (0, 1, 2)] = 1
        proj_scr[:, :, 3] = (rot_coord[:, 1, :, 2] - rot_coord[:, 2, :, 1])
        proj_scr[:, :, 4] = (rot_coord[:, 2, :, 0] - rot_coord[:, 0, :, 2])
        proj_scr[:, :, 5] = (rot_coord[:, 0, :, 1] - rot_coord[:, 1, :, 0])
        proj_scr *= np.sqrt(mol_weights)[:, None, None]
        proj_scr.shape = (-1, 6)
        proj_scr /= np.linalg.norm(proj_scr, axis=0)
        return proj_scr
    
    @property
    def proj_inv(self):
        if self._proj_inv is NotImplemented:
            natm, proj_scr, theta = self.natm, self.proj_scr, self.theta
            proj_inv = np.zeros((natm * 3, natm * 3))
            proj_inv[:, :6] = proj_scr
            cur = 6
            for i in range(0, natm * 3):
                vec_i = np.einsum("Ai, i -> A", proj_inv[:, :cur], proj_inv[i, :cur])
                vec_i[i] -= 1
                if np.linalg.norm(vec_i) > 1e-8:
                    proj_inv[:, cur] = vec_i / np.linalg.norm(vec_i)
                    cur += 1
                if cur >= natm * 3:
                    break
            proj_inv = proj_inv[:, 6:]
            self._proj_inv = proj_inv
        return self._proj_inv
    
    def _get_freq_qdiag(self):
        natm, proj_inv, theta, mol_weights = self.natm, self.proj_inv, self.theta, self.mol_weights
        e, q = np.linalg.eigh(proj_inv.T @ theta @ proj_inv)
        freq = np.sqrt(np.abs(e * E_h * 1000 * N_A / a_0**2)) / (2 * np.pi * c_0 * 100) * ((e > 0) * 2 - 1)
        self._freq = freq
        q_unnormed = np.einsum("AtQ, A -> AtQ", (proj_inv @ q).reshape(natm, 3, (proj_inv @ q).shape[-1]), 1 / np.sqrt(mol_weights))
        q_unnormed = q_unnormed.reshape(-1, q_unnormed.shape[-1])
        q_normed = q_unnormed / np.linalg.norm(q_unnormed, axis=0)
        return q_unnormed, q_normed
    
    @property
    def freq(self):
        if self._freq is NotImplemented:
            self._get_freq_qdiag()
        return self._freq
    
    @property
    def q(self):
        if self._q is NotImplemented:
            self._q, self._qnorm = self._get_freq_qdiag()
        return self._q
    
    @property
    def qnorm(self):
        if self._qnorm is NotImplemented:
            self._q, self._qnorm = self._get_freq_qdiag()
        return self._qnorm
