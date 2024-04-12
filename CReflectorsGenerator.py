import numpy as np
from enum import Enum
from scipy.signal import chirp
import scipy
import params


# converting target location r,theta,phi to r,abs(theta) for each bat ear
def target_loc_to_2ears(Dist, theta, phi):
    if theta < 0:  # [-pi:pi] -> [0:2*pi]
        theta = theta + 2 * np.pi
    phi = -phi + np.pi / 2  # [-pi/2:pi/2] -> [0:pi]

    x = Dist * np.sin(phi) * np.cos(theta)
    y = Dist * np.sin(phi) * np.sin(theta)
    z = Dist * np.cos(phi)

    y_left = y + params.DIST_EARS / 2
    y_right = y - params.DIST_EARS / 2

    y_fixed = np.array([y_left, y_right])

    Dist_fixed = np.sqrt(x ** 2 + y_fixed ** 2 + z ** 2)
    theta_fixed = np.arctan2(y_fixed, x)
    phi_fixed = np.arccos(z / Dist_fixed)

    for i in range(2):
        if theta_fixed[i] > np.pi:
            theta_fixed[i] = theta_fixed[i] - 2 * np.pi
    atheta_fixed = np.abs(theta_fixed)
    aphi_fixed = np.abs(-phi_fixed + np.pi / 2)
    angle_fixed = np.arccos(np.cos(atheta_fixed) * np.cos(aphi_fixed))

    return Dist_fixed, atheta_fixed


# converting target location from 2 ears in r,abs(theta) to general r,theta,phi to r,abs(theta)
def ears_to_target_loc(Dist_ar, theta_ar):
    Dist_min = np.min(Dist_ar)
    Dist_max = np.max(Dist_ar)

    # calculating r from center between the ears
    if Dist_max == Dist_min + params.DIST_EARS:
        Dist = Dist_max - params.DIST_EARS / 2
    else:
        Dist = np.sqrt(0.5 * np.sum(Dist_ar ** 2) - params.DIST_EARS ** 2 / 4)

    # calculating y location of target relative to center between the ears
    y_dist = (Dist_ar[0] ** 2 - Dist_ar[1] ** 2) / (2 * params.DIST_EARS)

    # calculating possible thetas from given abs(theta)
    theta_ar_opts = np.array(theta_ar)
    theta_ar_opts = np.vstack((theta_ar_opts, [-theta_ar[0], theta_ar[1]]))
    theta_ar_opts = np.vstack((theta_ar_opts, [theta_ar[0], -theta_ar[1]]))
    theta_ar_opts = np.vstack((theta_ar_opts, [-theta_ar[0], -theta_ar[1]]))
    len_opts = 4

    # for each possible theta combination calculate r, theta, phi, y
    res_ar = np.zeros((len_opts, 4))
    for k in range(len_opts):
        theta_ar1 = np.copy(theta_ar_opts[k])

        for i in range(2):
            if theta_ar1[i] < 0:  # [-pi:pi] -> [0:2*pi]
                theta_ar1[i] = theta_ar1[i] + 2 * np.pi

        tan_theta_left = np.tan(theta_ar1[0])
        tan_theta_right = np.tan(theta_ar1[1])

        x = (((y_dist + params.DIST_EARS / 2) / tan_theta_left) +
             ((y_dist - params.DIST_EARS / 2) / tan_theta_right)) / 2
        y = y_dist
        z = np.sqrt(Dist ** 2 - x ** 2 - y ** 2)

        phi = np.arccos(z / Dist)
        theta = np.arctan2(y / np.sin(phi), x / np.sin(phi))

        if theta > np.pi:  # [0:2*pi] -> [-pi:pi]
            theta = theta - 2 * np.pi

        phi = -phi + np.pi / 2  # [0:pi] -> [-pi/2:pi/2]

        res_ar[k, 0] = Dist
        res_ar[k, 1] = theta
        res_ar[k, 2] = phi
        res_ar[k, 3] = y

    # choose the option which best reconstructs the initial abs(theta), y, r
    diff = np.array([])
    for i in range(len_opts):
        Dist_fixed, atheta_fixed = target_loc_to_2ears(res_ar[i, 0], res_ar[i, 1], res_ar[i, 2])
        diff = np.append(diff, (res_ar[i, 3] - y_dist) ** 2 + np.sum((atheta_fixed - np.abs(theta_ar_opts[i // 2])) ** 2) + np.sum((Dist_fixed - Dist_ar) ** 2))

    res = res_ar[np.argmin(diff)]

    Dist = res[0]
    theta = res[1]
    phi = res[2]
    y = res[3]

    return Dist, theta, phi

class CReflectorsGenerator:
    def __init__(self, max_targets):
        self.max_targets = max_targets
        self.prev_echoes: np.ndarray = np.zeros(self.max_targets)
        self.prev_echoes = np.vstack((self.prev_echoes, np.zeros(self.max_targets)))
        self.base_noise = 0
        self.ref_eval = False
        pass

    # restore class to initial conditions
    def clean(self):
        self.prev_echoes = np.zeros(self.max_targets)
        self.prev_echoes = np.vstack((self.prev_echoes, np.zeros(self.max_targets)))
        self.base_noise = 0
        self.ref_eval = False

    # calculating gain ratio as function of target phi location
    def G_G0_calc(self, k, a, phi):
        if phi == 0 or phi == np.pi:
            G_G0 = np.ones_like(k)
        else:
            var = k * a * np.sin(phi)
            G_G0 = (2 * scipy.special.j1(var) / var) ** 2

        return G_G0

    # calculating the gain of Rx signal
    def gain_calc(self, phi, f, Dist, R_target):
        V_p = params.V_P  # m/sec
        lam = V_p / f  # m

        a_mouth = params.A_MOUTH  # m, mouth length
        a_ear = params.A_EAR  # m, ear length
        k = 2 * np.pi / lam  # 1/m

        G0_Tx = 1
        G0_Rx = (2 * np.pi * a_ear / lam) ** 2

        G_Tx = G0_Tx * self.G_G0_calc(k, a_mouth, phi)
        G_Rx = G0_Rx * self.G_G0_calc(k, a_ear, phi)

        gain_back_hem = np.ones_like(f)
        if np.abs(phi) > np.pi / 2:
            gain_back_hem = gain_back_hem * ((-1.98 * np.abs(phi) / np.pi) + 1.99)

        G_Tx = G_Tx * gain_back_hem  # with back_hem
        G_Rx = G_Rx * gain_back_hem  # with back_hem

        alpha_att = (3.8e-2) * (f / 1000) - 0.3  # dB/m, atmospheric absorption
        RCS = np.pi * R_target ** 2  # disc of radius R_target for targets larger then

        gain = ((G_Tx * G_Rx * lam ** 2) / (((4 * np.pi) ** 3) * Dist ** 4)) * \
               RCS  # without back_hem and atmospheric absorption
        gain = gain * 10 ** (-2 * alpha_att * (Dist - 0.1) / 10)

        return gain

    # calculating the Rx signal for a specific scene
    def create_reflectors_waveform_3D(self, range_list_in_meters: np.ndarray, theta_list: np.ndarray,
                                      phi_list: np.ndarray, R_list_in_meters: np.ndarray):
        data_valid = True

        # input validation
        equal_len = np.shape(theta_list) == np.shape(range_list_in_meters)
        equal_len = equal_len and np.shape(phi_list) == np.shape(range_list_in_meters)
        equal_len = equal_len and np.shape(R_list_in_meters) == np.shape(range_list_in_meters)
        assert equal_len, "size of inputs must be the same"

        list_size = np.size(range_list_in_meters)
        more_than_max_targets = list_size <= self.max_targets
        assert more_than_max_targets

        # based on sample frequency
        sample_freq_hz = params.F_SAMPLE
        # sound speed in air m / s
        Vp = params.V_P
        Tx_power = params.TX_SIGNAL_POWER  # not dB
        N_power = params.NOISE_POWER  # not dB

        # range uniqueness constraint
        max_decimals = params.NUM_DECIMALS_ROUND

        # range rounding
        range_list_in_meters_rounded = np.around(range_list_in_meters,
                                                 decimals=max_decimals)
        range_list_in_meters_orig = range_list_in_meters
        range_list_in_meters = range_list_in_meters_rounded

        # check if all ranges are unique
        range_list_in_meters_unique = np.unique(range_list_in_meters)
        if not (len(range_list_in_meters_unique) == len(range_list_in_meters)):
            data_valid = False

        # sort ranges and appropriate theta, phi, R
        inds_sort = np.argsort(range_list_in_meters)
        range_list_in_meters = range_list_in_meters[inds_sort]
        theta_list = theta_list[inds_sort]
        phi_list = phi_list[inds_sort]
        R_list_in_meters = R_list_in_meters[inds_sort]

        # translating scene parameters for each ear relevant parameters
        Dist_fixed_ar = np.zeros((list_size, 2))
        angle_fixed_ar = np.zeros((list_size, 2))
        for i in range(list_size):
            Dist = range_list_in_meters_orig[i]
            theta = theta_list[i]
            phi = phi_list[i]

            Dist_fixed, angle_fixed = target_loc_to_2ears(Dist, theta, phi)

            Dist_fixed_ar[i, :] = Dist_fixed
            angle_fixed_ar[i, :] = angle_fixed

        # radar range equation R = 0.5 * Vp * Delay
        delays_sec = Dist_fixed_ar * 2 / Vp

        # translate to echo locations in sample domain
        echo_locations = np.round(delays_sec * sample_freq_hz).astype(int).T - 1

        # check uniqueness of echo locations
        echo_locations_unique_left = np.unique(echo_locations[0, :])
        echo_locations_unique_right = np.unique(echo_locations[1, :])
        if not (len(echo_locations_unique_left) == len(echo_locations[0, :])):
            data_valid = False

        if not (len(echo_locations_unique_left) == len(echo_locations_unique_right)):
            data_valid = False

        # append zeros in order to add to numpy array
        padding = self.max_targets - np.size(echo_locations, axis=1)
        pad = np.zeros((2, padding))
        echo_locations_padded = np.hstack((echo_locations, pad))

        # check if we generated this range before, data should be unique for training
        if np.shape(self.prev_echoes)[0] > 0 and data_valid:
            is_in_list = np.any(np.all(np.equal(self.prev_echoes, echo_locations_padded[0, :]), axis=1))
            is_in_list1 = np.any(np.all(np.equal(self.prev_echoes, echo_locations_padded[1, :]), axis=1))
            if is_in_list or is_in_list1:
                data_valid = False
            else:
                self.prev_echoes = np.vstack((self.prev_echoes, echo_locations_padded))
                pass

        # project assumption
        max_range_meters = params.DISTANCE_MAX + params.DIST_EARS / 2
        max_delay_sec = 2 * max_range_meters / Vp

        # Tx signal properties
        signal_time_sec = params.TX_SIGNAL_LENGTH
        fmin_hz = params.F_CHIRP_START
        fmax_hz = params.F_CHIRP_END
        tx_time_sec_vec = np.arange(0, signal_time_sec, 1 / sample_freq_hz)

        # signal model is pulsed lfm down, single pulse assumption
        tx_signal = chirp(tx_time_sec_vec, fmin_hz, signal_time_sec, fmax_hz, method='linear')
        f_vec = np.linspace(fmin_hz, fmax_hz, len(tx_signal))

        # reflection  model
        delay_vec_len = int(sample_freq_hz * max_delay_sec)
        echoes_no_noise = np.zeros((2, params.DATA_LEN_3D))

        # superposition of echos to create Rx signal
        for i in range(list_size):
            Echo_deltas = np.zeros((2, delay_vec_len))
            R_target = R_list_in_meters[i]

            Dist_left = Dist_fixed_ar[i, 0]
            angle_left = angle_fixed_ar[i, 0]

            Dist_right = Dist_fixed_ar[i, 1]
            angle_right = angle_fixed_ar[i, 1]

            rx_signal_left = tx_signal * self.gain_calc(angle_left, f_vec, Dist_left, R_target)
            rx_signal_right = tx_signal * self.gain_calc(angle_right, f_vec, Dist_right, R_target)

            if data_valid:
                Echo_deltas[0, echo_locations[0, i]] = 1
                Echo_deltas[1, echo_locations[1, i]] = 1

            # conv with discrete delta functions
            echo_no_noise_left = Tx_power * np.convolve(Echo_deltas[0, :], rx_signal_left)
            echo_no_noise_right = Tx_power * np.convolve(Echo_deltas[1, :],
                                                         rx_signal_right)

            echo_no_noise = np.vstack((echo_no_noise_left, echo_no_noise_right))
            echo_no_noise = np.hstack((echo_no_noise, np.zeros((2, 1))))

            echoes_no_noise = echoes_no_noise + echo_no_noise

        Echo_deltas = np.zeros((2, delay_vec_len))

        Echo_deltas[0, echo_locations[0, :list_size]] = 1
        Echo_deltas[1, echo_locations[1, :list_size]] = 1

        # adding white noise
        echoes = echoes_no_noise + np.random.normal(0, N_power, (2, params.DATA_LEN_3D))

        # classify scene invalid if SNR<4 for at least one echo after superposition
        for echo_locs_ear, echo_ear in zip(echo_locations, echoes):
            for echo_loc in echo_locs_ear:
                if np.max(np.abs(echo_ear[echo_loc:echo_loc + params.TX_SIGNAL_LENGTH_IND])) < 4 * N_power:
                    data_valid = False

        # create segmentation mask
        tx_of_ones = np.ones_like(tx_signal)
        seg_mask = np.convolve(Echo_deltas[0, :], tx_of_ones)
        seg_mask = np.vstack((seg_mask, np.convolve(Echo_deltas[1, :], tx_of_ones)))

        # append zeroes to range_list_in_meters if required, zero means no target
        zero_to_append = self.max_targets - list_size
        range_list_in_meters_fixed_size = np.pad(range_list_in_meters,
                                                 (0, zero_to_append), 'constant')
        theta_list_fixed_size = np.pad(theta_list, (0, zero_to_append), 'constant')
        phi_list_fixed_size = np.pad(phi_list, (0, zero_to_append), 'constant')
        R_list_fixed_size = np.pad(R_list_in_meters, (0, zero_to_append), 'constant')

        seg_mask = np.hstack((seg_mask, np.zeros((2, 1))))

        return tx_signal, echoes, range_list_in_meters_fixed_size, theta_list_fixed_size, phi_list_fixed_size, R_list_fixed_size, seg_mask, data_valid
