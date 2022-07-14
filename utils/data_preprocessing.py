import uproot
import awkward as ak
import tensorflow as tf
import numpy as np
from hydra.utils import to_absolute_path

def load_from_file(file_name, tree_name, input_branches):
    print(f'      - {file_name}')
    
    # open ROOT file and retireve awkward arrays
    with uproot.open(to_absolute_path(f'{file_name}.root')) as f:
        a = f[tree_name].arrays(input_branches, how='zip')

    return a

def awkward_to_ragged(a, particle_type, feature_names):
    pf_lengths = ak.count(a[particle_type, feature_names[0]], axis=1)
    ragged_pf_features = []
    for feature in feature_names:
        pf_a = ak.flatten(a[particle_type, feature])
        pf_a = ak.values_astype(pf_a, np.float32)
        ragged_pf_features.append(tf.RaggedTensor.from_row_lengths(pf_a, pf_lengths))
    ragged_pf = tf.stack(ragged_pf_features, axis=-1)
    return ragged_pf

def preprocess_array(a):
    # remove taus with abnormal phi
    a = a[np.abs(a['tau_phi'])<2.*np.pi] 

    # ------- PF candidates ------- #

    # shift delta phi into [-pi, pi] range 
    dphi_array = (a['pfCand', 'phi'] - a['tau_phi'])
    dphi_array = np.where(dphi_array <= np.pi, dphi_array, dphi_array - 2*np.pi)
    dphi_array = np.where(dphi_array >= -np.pi, dphi_array, dphi_array + 2*np.pi)

    # add features
    a['pfCand', 'dphi'] = dphi_array
    a['pfCand', 'deta'] = a['pfCand', 'eta'] - a['tau_eta']
    a['pfCand', 'rel_pt'] = a['pfCand', 'pt'] / a['tau_pt']
    a['pfCand', 'r'] = np.sqrt(np.square(a['pfCand', 'deta']) + np.square(a['pfCand', 'dphi']))
    a['pfCand', 'theta'] = np.arctan2(a['pfCand', 'dphi'], a['pfCand', 'deta']) # dphi -> y, deta -> x
    a['pfCand', 'particle_type'] = a['pfCand', 'particleType'] - 1

    # vertices
    a['pfCand', 'vertex_dx'] = a['pfCand', 'vertex_x'] - a['pv_x']
    a['pfCand', 'vertex_dy'] = a['pfCand', 'vertex_y'] - a['pv_y']
    a['pfCand', 'vertex_dz'] = a['pfCand', 'vertex_z'] - a['pv_z']
    a['pfCand', 'vertex_dx_tauFL'] = a['pfCand', 'vertex_x'] - a['pv_x'] - a['tau_flightLength_x']
    a['pfCand', 'vertex_dy_tauFL'] = a['pfCand', 'vertex_y'] - a['pv_y'] - a['tau_flightLength_y']
    a['pfCand', 'vertex_dz_tauFL'] = a['pfCand', 'vertex_z'] - a['pv_z'] - a['tau_flightLength_z']

    # IP, track info
    has_track_details = a['pfCand', 'hasTrackDetails'] == 1
    has_track_details_track_ndof = has_track_details * (a['pfCand', 'track_ndof'] > 0)
    a['pfCand', 'dxy'] = ak.where(has_track_details, a['pfCand', 'dxy'], 0)
    a['pfCand', 'dxy_sig'] = ak.where(has_track_details, np.abs(a['pfCand', 'dxy'])/a['pfCand', 'dxy_error'], 0)
    a['pfCand', 'dz'] = ak.where(has_track_details, a['pfCand', 'dz'], 0)
    a['pfCand', 'dz_sig'] = ak.where(has_track_details, np.abs(a['pfCand', 'dz'])/a['pfCand', 'dz_error'], 0)
    a['pfCand', 'track_ndof'] = ak.where(has_track_details_track_ndof, a['pfCand', 'track_ndof'], 0)
    a['pfCand', 'chi2_ndof'] = ak.where(has_track_details_track_ndof, a['pfCand', 'track_chi2']/a['pfCand', 'track_ndof'], 0)

    # ------- Electrons ------- #
    
    # shift delta phi into [-pi, pi] range 
    dphi_array = (a['ele', 'phi'] - a['tau_phi'])
    dphi_array = np.where(dphi_array <= np.pi, dphi_array, dphi_array - 2*np.pi)
    dphi_array = np.where(dphi_array >= -np.pi, dphi_array, dphi_array + 2*np.pi)

    a['ele', 'dphi'] = dphi_array
    a['ele', 'deta'] = a['ele', 'eta'] - a['tau_eta']
    a['ele', 'rel_pt'] = a['ele', 'pt'] / a['tau_pt']
    a['ele', 'r'] = np.sqrt(np.square(a['ele', 'deta']) + np.square(a['ele', 'dphi']))
    a['ele', 'theta'] = np.arctan2(a['ele', 'dphi'], a['ele', 'deta']) # dphi -> y, deta -> x
    a['ele', 'particle_type'] = 7 # assuming PF candidate types are [0..6]

    ele_cc_valid = a['ele', 'cc_ele_energy'] >= 0
    a['ele', 'cc_valid'] = ele_cc_valid
    a['ele', 'cc_ele_rel_energy'] = ak.where(ele_cc_valid, a['ele', 'cc_ele_energy']/a['ele', 'pt'], 0)
    a['ele', 'cc_gamma_rel_energy'] = ak.where(ele_cc_valid, a['ele', 'cc_gamma_energy']/a['ele', 'cc_ele_energy'], 0)
    a['ele', 'cc_n_gamma'] = ak.where(ele_cc_valid, a['ele', 'cc_n_gamma'], -1)
    a['ele', 'rel_trackMomentumAtVtx'] = a['ele', 'trackMomentumAtVtx']/a['ele', 'pt']
    a['ele', 'rel_trackMomentumAtCalo'] = a['ele', 'trackMomentumAtCalo']/a['ele', 'pt']
    a['ele', 'rel_trackMomentumOut'] = a['ele', 'trackMomentumOut']/a['ele', 'pt']
    a['ele', 'rel_trackMomentumAtEleClus'] = a['ele', 'trackMomentumAtEleClus']/a['ele', 'pt']
    a['ele', 'rel_trackMomentumAtVtxWithConstraint'] = a['ele', 'trackMomentumAtVtxWithConstraint']/a['ele', 'pt']
    a['ele', 'rel_ecalEnergy'] = a['ele', 'ecalEnergy']/a['ele', 'pt']
    a['ele', 'ecalEnergy_sig'] = a['ele', 'ecalEnergy']/a['ele', 'ecalEnergy_error']
    a['ele', 'rel_gsfTrack_pt'] = a['ele', 'gsfTrack_pt']/a['ele', 'pt']
    a['ele', 'gsfTrack_pt_sig'] = a['ele', 'gsfTrack_pt']/a['ele', 'gsfTrack_pt_error']

    ele_has_closestCtfTrack = a['ele', 'closestCtfTrack_normalizedChi2'] >= 0
    a['ele', 'has_closestCtfTrack'] = ele_has_closestCtfTrack
    a['ele', 'closestCtfTrack_normalizedChi2'] = ak.where(ele_has_closestCtfTrack, a['ele', 'closestCtfTrack_normalizedChi2'], 0)
    a['ele', 'closestCtfTrack_numberOfValidHits'] = ak.where(ele_has_closestCtfTrack, a['ele', 'closestCtfTrack_numberOfValidHits'], 0)

    # preprocess NaNs
    a = ak.nan_to_num(a, nan=0., posinf=0., neginf=0.)

    return a 