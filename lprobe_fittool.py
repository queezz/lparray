"""
Fitting tool for Langmuir Probe Array analysis in NIFS

Author A. Kuzmin
e-mail: arseniy.a.kuzmin@gmail.com
April 2018
"""

import numpy as np

try:
    from labcom.Retriever import Retriever as rtv
except:
    print("failed to load labcom.Retriever")
import tools as tls
import matplotlib.pylab as plt

# MARK: Get Data


def get_raw_data(shot, **kwr):
    """Read raw data for the Lengmuir8 probe with retrieve
    ar = shot
    kwr = subshot, data_name, chnls
    """
    subshot = kwr.get("subshot", 1)
    data_name = kwr.get("data_name", "Langmuir8")
    chnls = kwr.get("chnls", [1])
    r = rtv()
    return r.get(data_name, shot, subshot, chnls)


def get_vp(shot, **kwr):
    """get voltage signals for given shot"""
    chnls = kwr.get("chnls", range(281, 286))  # get chnls if specified
    kwr.update([("chnls", chnls)])  # update kwr dictionary
    data = get_raw_data(shot, **kwr)
    out = dict()
    for j, i in enumerate(chnls):
        out[i] = data[j].val()
    out["t"] = data[0].time()
    return out


def get_ip(shot, **kwr):
    """get current signals for a given shot"""
    chnls = kwr.get("chnls", range(1, 281))
    kwr.update([("chnls", chnls)])
    data = get_raw_data(shot, **kwr)
    out = dict()
    for j, i in enumerate(chnls):
        out[i] = data[j].val()
    out["t"] = data[0].time()
    return out


# MARK: Calibrate


def calibrate_ip(data, **kwr):
    """
    calculate zero offset for given data
    data = dictionary of data, keys = channel numbers
    """
    multiplier = kwr.get("multiplyer", 3.0)
    zero_range = kwr.get("zero_range", [0, 0.1])
    fro = tls.find_nearest(data["t"], zero_range[0])
    tto = tls.find_nearest(data["t"], zero_range[-1])
    chnls = [i for i in data.keys() if not i == "t"]
    for ch in chnls:
        data[ch] += -data[ch][fro:tto].mean()
        data[ch] = data[ch] * multiplier
    return data


def calibrate_vp(data, shot, **kwr):
    """adjust offset, multiply signal for vp in"""
    prm = ch_info(shot)
    vp0_calibr = prm["vp0_calibr"]
    vp0_offset = prm["vp0_offset"]
    chnls = [i for i in data.keys() if not i == "t"]
    for ch in chnls:
        data[ch] = data[ch] - vp0_offset[ch]
        data[ch] = data[ch] * vp0_calibr[ch]
    return data


def process_vp(shot, **kwr):
    """get Vp data, calibrate, return"""
    data = get_vp(shot, **kwr)
    return calibrate_vp(data, shot, **kwr)


def process_ip(shot, **kwr):
    """get Ip data, calibrate, return"""
    data = get_ip(shot, **kwr)
    return calibrate_ip(data, shot, **kwr)


# MARK: Mode


def measurment_mode(data, **kwr):
    """Analize Voltage signal and return measurment mode of
    the Langmuir probes.
    0 = not biased.
    1 = Ion saturation current mode.
    2 = Ion saturation current + Volt-Ampere characteristic mode.
    data = {ch:signal, ...}
    """
    zero_range = kwr.get("zero_range", [0, 0.1])
    v_up = 50
    v_low = -50
    ratio = 0.2
    fro = tls.find_nearest(data["t"], zero_range[0])
    tto = tls.find_nearest(data["t"], zero_range[-1])
    chnls = [i for i in data.keys() if not i == "t"]
    mode = dict()
    for ch in chnls:
        y = data[ch][fro:tto]  # select data points around time = 0
        y_mean = y.mean()
        idx_up = np.where(y >= v_up + y_mean)
        idx_low = np.where(y >= v_low + y_mean)
        ratio_up = len(idx_up[0]) / float(len(y))
        ratio_low = len(idx_low[0]) / float(len(y))
        if ratio_up >= ratio and ratio_low >= ratio:
            mode[ch] = 2
        if ratio_up < ratio and ratio_low < ratio and y_mean < v_low:
            mode[ch] = 1
        if ratio_up < ratio and ratio_low < ratio and y_mean > v_low:
            mode[ch] = 1
    return mode


def plot_data(data, **kwr):
    """plot data from a dictionary of data"""
    chnls = [i for i in data.keys() if not i == "t"]
    for ch in chnls:
        plt.plot(data["t"], data[ch])


# MARK: Ch. Info


def ch_info(shot):
    """Parameters (prm) of the channel connection and positions on the
    tiles; basic measurment settings.
    ip_ch = list of current channels for tiles 2L, 2R and so on
    tiles = tiles names, where probes are installed
    vp0_offset = voltage offset for voltage channels
    vp0_calibr = calibration factor to convert Vp signal to Volts
    """
    prm = dict()
    tiles = [
        "2L",
        "2R",
        "4L",
        "4R",
        "6L",
        "6R",
        "7L",
        "7R",
        "8L",
        "8R",
        "9L",
        "9R",
        "10L",
        "10R",
    ]
    prm["tiles"] = tiles

    ip_ch = dict()
    ip_ch["2L"] = range(1, 21)  # from 19th cycle
    ip_ch["2R"] = range(21, 41)  # from 19th cycle
    ip_ch["4L"] = range(81, 101)
    ip_ch["4R"] = range(101, 121)
    ip_ch["6L"] = range(121, 141)
    ip_ch["6R"] = (
        [146, 147, 150, 144, 145]
        + list(range(141, 144))
        + [159, 160]
        + list(range(153, 159))
        + [151, 152]
    )
    ip_ch["7L"] = range(161, 181)
    ip_ch["7R"] = range(181, 201)
    ip_ch["8L"] = range(201, 221)
    ip_ch["8R"] = range(221, 241)
    ip_ch["9L"] = range(41, 61)
    ip_ch["9R"] = range(61, 81)
    ip_ch["10L"] = range(241, 261)
    ip_ch["10R"] = range(261, 281)
    prm["ip_ch"] = ip_ch

    prm["vp0_ch"] = [
        281,
        281,
        281,
        281,
        282,
        282,
        283,
        283,
        282,
        282,
        285,
        284,
        283,
        283,
    ]
    # Vp offset for each tile (befor multiplication)
    vp0_offset = dict()
    vp0_offset[281] = 0.0071
    vp0_offset[282] = 0.0067
    vp0_offset[283] = 0.0062
    vp0_offset[284] = 0.0076
    vp0_offset[285] = 0.0070
    vp0_offset[286] = 0.0075
    prm["vp0_offset"] = vp0_offset

    vp0_calibr = dict()
    vp0_calibr[281] = 30 * 30
    vp0_calibr[282] = 30 * 30
    vp0_calibr[283] = 30 * 30
    vp0_calibr[284] = 101 * 30
    vp0_calibr[285] = 101 * 30
    vp0_calibr[286] = 101 * 30
    if shot <= 131165:
        vp0_calibr[284] = 30  # line missing
        vp0_calibr[285] = 30  # line missing
    prm["vp0_calibr"] = vp0_calibr

    return prm


# MARK: SAW FIT


def saw_voltage(x, amplitude=1, omega=1.0, phase=0, skew=0.5, baseline=0):
    """saw-like signal for fitting the voltage for LPA"""
    from scipy.signal import sawtooth

    return amplitude * sawtooth(omega * np.pi * x + np.pi * phase, skew) + baseline


def fit_saw(x, y):
    """fit saw-like voltage for Langmuir Probes"""
    from lmfit import Model

    model = Model(saw_voltage)
    params = model.make_params()
    params["amplitude"].set(145, min=100, max=200)
    params["skew"].set(0.5, vary=False)
    params["omega"].set(500, min=499, max=501)
    params["baseline"].set(-45, min=-200, max=200)
    params["phase"].set(0, min=-1, max=1, vary=True)

    result = model.fit(y, params, x=x)
    return result


# MARK: Extract Data


def get_rampups(x, y):
    """Calculate the mask for the voltage ramp-ups"""
    # 1. get the sawtooth waveform
    result = fit_saw(x, y)

    # 2. use the sawtooth to get rising voltage parts of the signal
    names = ["amplitude", "omega", "phase", "skew", "baseline"]
    coefs = [result.params[name].value for name in names]
    xx = np.linspace(
        x.min(), x.max(), len(x)
    )  # just to make sure that the step is constant
    yy = saw_voltage(xx, *coefs)
    zz = np.gradient(
        yy,
    )
    ind = np.where(zz > 0)[0]  # select the positive derivative
    # 2.1 get the start-end indexes of the voltage ramp-up
    j = [0]
    for i, _ in enumerate(ind[:-1]):
        if ind[i] + 1 < ind[i + 1]:
            j.append(i)
    j.append(len(ind))
    # 2.2 create a list of indexes for all the ramp-ups in the given timeframe
    ind1 = [ind[range(j[i] + 1, j[i + 1])] for i, _ in enumerate(j[:-1])]

    return {"mask": ind1, "saw_coeff": coefs}


def get_voltage(shot, t_start, t_end, **kws):
    """Get voltage data for the given probes
    and calculate mask for voltage ramp-ups
    """
    vps = get_vp(shot)
    vp = calibrate_vp(vps, shot)
    vt = vp["t"]
    vch = kws.get("vch", 281)
    volt = vp[vch]

    # select data in the given time frame
    ind = np.where((vt >= t_start) & (vt < t_end))
    vt = vt[ind]
    volt = volt[ind]

    # mask data, select only ramp-ups inside the [t0,t1] interval
    rampups = get_rampups(vt, volt)
    rampup_mask = rampups["mask"]
    saw_coeff = rampups["saw_coeff"]
    res = {"v": volt, "mask": rampup_mask, "t": vt, "saw_coeff": saw_coeff}
    return res


def get_current(shot, channel, t_start, t_end, **kws):
    """Get current for a given shot and a given channel
    Subtract inducted current, crop to the given time interval
    """
    # read current data for a given channel
    ips = get_vp(shot, chnls=[channel])
    ip = calibrate_ip(ips)

    cur = ip[channel]
    ct = ip["t"]

    # Subtract the inducted current background
    msk = np.where((ct > 0) & (ct < 0.01))

    level = cur[:1000].mean()
    offset_mask = np.where(cur[:4000] - level < 0)
    current_offset = cur[offset_mask].mean()
    cur = cur - level - current_offset

    ind = np.where((ct >= t_start) & (ct < t_end))
    ct = ct[ind]
    cur = cur[ind]
    # res = {'current':cur,'t':ct}
    return cur


# MARK: LP Fit


def prob_func(x, iis, ies, vs, te, slope):
    """Function for Langmuir Probe data fitting"""
    # iis,ies,vs,te,slope = args
    return slope * (x - vs) + iis + ies * np.exp((x - vs) / te)
    # return iis + ies*np.exp((x-vs)/te)


def fit_lp(x, y, **kws):
    """fit Langmuir Probe I(V) characteristic"""
    from lmfit import Model

    model = Model(prob_func)
    te_max = kws.get("te_max", 20.0)
    te_min = kws.get("te_min", 4.0)
    te_init = kws.get("te_init", 10.0)
    iis_min = kws.get("iis_min", 5.0e-4)
    iis_max = kws.get("iis_max", 1.0e2)
    iis_init = kws.get("iis_init", 0.5)
    params = model.make_params()
    params["iis"].set(iis_init, min=iis_min, max=iis_max)
    params["ies"].set(-10, min=-300, max=0)
    params["vs"].set(0)
    params["te"].set(te_init, min=te_min, max=te_max)
    params["slope"].set(0, min=-1, max=1)

    result = model.fit(y, params, x=x, method=kws.get("method", "lbfgsb"))

    return result


def movav(x, y, win):
    """moving aveerage
    Calculate an average for indexes in win [i*win:(i+1)*win] and move
    """
    windpw = int(win)
    n = int(len(y) / win)
    y1 = [np.mean(y[win * i : win * (i + 1)]) for i in range(n)]
    x1 = [np.mean(x[win * i : win * (i + 1)]) for i in range(n)]
    return x1, y1


# MARK: Step Fit


def step_fit_lp(v_step_raw, i_step_raw, **kws):
    """Fit one ramp-up region of the LPA.
    vres = {'v':voltage,'t':time,....} full dictionary
    cur = [current]
    mask = mask for one ramp-up

    UPDATE no mask provided, instead
    i_step_raw and v_ste_raw are provided
    """
    vmax = kws.get("vmax", 15)
    dv = kws.get("dv", 30.0)
    dry_run = kws.get("dry_run", False)  # return masked data, no fit

    # 2. filter out noisy voltage values
    by_scatter = kws.get("by_scatter", False)
    by_deriv = kws.get("by_deriv", True)

    if by_scatter:
        saw_coeff = kws.get("saw_coeff")
        v_saw = saw_voltage(t_step_raw, *saw_coeff)
        mask_vnoise = np.where(abs(v_saw - v_step_raw) < dv)

    if by_deriv:
        deriv = np.gradient(v_step_raw)
        ind000 = np.where(np.abs(deriv) > dv)
        if len(ind000[0]) > 0:
            mask_vnoise = np.arange(np.min(ind000[0]))
        else:
            mask_vnoise = np.arange(len(v_step_raw))

    # t_step_filtered = t_step_raw[mask_vnoise]
    v_step_filtered = v_step_raw[mask_vnoise]
    i_step_filtered = i_step_raw[mask_vnoise]

    # 2.1 Reduce number of data points in the negative voltage partition
    # and replace them with moving average instead. To increase reliability
    # of the fit
    if 0:
        ng_mask = np.where(v_step_filtered < -10)
        ps_mask = np.where(v_step_filtered >= -10)
        x1, y1 = movav(v_step_filtered[ng_mask], i_step_filtered[ng_mask], 10)
        v_step_filtered = np.append(x1, v_step_filtered[ps_mask])
        i_step_filtered = np.append(y1, i_step_filtered[ps_mask])

        ng_mask = np.where(v_step_filtered < 10)
        ps_mask = np.where(v_step_filtered >= 10)
        x1, y1 = movav(v_step_filtered[ps_mask], i_step_filtered[ps_mask], 3)
        v_step_filtered = np.append(v_step_filtered[ng_mask], x1)
        i_step_filtered = np.append(i_step_filtered[ng_mask], y1)

    # 3. cut off points where voltage is higher than vmax
    # print np.max(v_step_filtered), vmax
    vmax_mask = np.where(v_step_filtered < vmax)
    v_f = v_step_filtered[vmax_mask]
    i_f = i_step_filtered[vmax_mask]

    # 4. Do the FIT
    fit_params = {
        "te_max": kws.get("te_max", 20.0),
        "te_min": kws.get("te_min", 4.0),
        "te_init": kws.get("te_init", 10.0),
        "iis_min": kws.get("iis_min", 5.0e-4),
        "iis_max": kws.get("iis_max", 1.0e2),
        "iis_init": kws.get("iis_init", 0.5),
        "method": kws.get("method", "lbfgsb"),
    }
    try:
        if not dry_run:
            result = fit_lp(v_f, i_f, **fit_params)
            fit_success = True
        else:
            fit_success = False
    except:
        fit_success = False

    # 5.a Prepare result to pass over
    if not dry_run and fit_success:
        v_even = np.linspace(np.min(v_f), np.max(v_f), len(v_f))
        names = ["iis", "ies", "vs", "te", "slope"]
        coefs = [result.params[name].value for name in names]
        i_even = prob_func(v_even, *coefs)

        # 5.a.1 self-defined chisq
        y1 = prob_func(v_f, *coefs)
        ind0 = np.where(v_f > -10)
        y2 = i_f[ind0]
        y3 = y1[ind0]
        my_chi = np.sum((y3 - y2) ** 2)

    # 5.b Return masked data for plotting, no fit reults
    if dry_run or not fit_success:
        v_even = np.empty(len(v_f))
        i_even = np.empty(len(i_f))
        v_even[:] = np.nan
        i_even[:] = np.nan
        result = None
        my_chi = np.nan

    res = {
        "v_f": v_f,
        "i_f": i_f,
        "v_even": v_even,
        "fit_success": fit_success,
        "i_even": i_even,
        "result": result,
        "mask_vnoise": mask_vnoise,
        "my_chi": my_chi,
    }
    return res


if __name__ == "__main__":
    pass
