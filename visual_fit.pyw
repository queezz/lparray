#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Fitting tool for Langmuir Probe Array analysis in NIFS
Visual conformation of the fit quality for each voltage ramp-up phase based on pyqtgraph module.

Author A. Kuzmin
e-mail: arseniy.a.kuzmin@gmail.com
April 2018
"""

from __future__ import print_function
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
import functools

try:
    import Queue as que  # for threading.
except:
    import queue as que

from scipy.signal import sawtooth
from scipy.constants import m_p, elementary_charge
import numpy as np

# import datetime
import time
import os
import sys

from lmfit import Model

errmess = ""
try:
    from labcom.Retriever import Retriever as rtv
except:
    print("labcom module not found")
    errmess = "labcom module not found"

import lprobe_fittool as lp

# change the stdout output from console(terminal) to this application
# True - this app, False - release it, put it back to console(terminal)
# if True - a log file will also be saved with all stdout
CAPTURE_STD = False

# == == == == == == == == == == == == == == == == == == == == == == == == ==
# GUI description. Create main window and widgets with plots and controls
# == == == == == == == == == == == == == == == == == == == == == == == == ==

import ctypes
import platform

myappid = "mycompany.myproduct.subproduct.version"  # arbitrary string
app = QtGui.QApplication([])

# Set-up icon for this application
try:
    app.setWindowIcon(QtGui.QIcon("icons/langmuir_tool.png"))
except:
    pass

if platform.system() == "Windows":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

win = QtGui.QMainWindow()
tabwidg = QtGui.QTabWidget()
area = DockArea()
area1 = DockArea()
tabwidg.addTab(area, "Main program")
tabwidg.addTab(area1, "Channel selection table")
# win.setCentralWidget(area)
win.setCentralWidget(tabwidg)
win.resize(1000, 700)
win.setWindowTitle("Langmuir Probe Array Fitting Tool")
win.setAcceptDrops(True)
# center window
resolution = QtGui.QDesktopWidget().screenGeometry()
win.move(
    (resolution.width() / 2) - (win.frameSize().width() / 2),
    (resolution.height() / 2) - (win.frameSize().height() / 2),
)

win.statusBar().showMessage("".format(errmess))
# == == == == == == == == == == == == == == == == == == == == == == == == ==
def gui_setup_spinbox(spbox, val, min, max):
    """set up spinbox value, min and max"""
    spbox.setMinimum(min)
    spbox.setMaximum(max)
    spbox.setValue(val)


# == == == == == == == == == == == == == == == == == == == == == == == == ==
d_plots = Dock("Plots", size=(300, 300))
# d_plots.setAcceptDrops(True)
d_load = Dock("Shot parameters", size=(100, 100))
d_controls = Dock("Fit controls", size=(200, 300))
d_results = Dock("Fit Results", size=(100, 300))  # plots
d_constr = Dock("Fit constrains", size=(10, 100))
d_cmd = Dock("CMD out", size=(300, 300))  # write the cmd output to this dock
d_chnl = Dock("Channels", size=(100, 100))
area.addDock(d_cmd)
area.addDock(d_plots, "above", d_cmd)
area.addDock(d_results, "bottom", d_plots)
area.addDock(d_load, "right")
area.addDock(d_controls, "bottom", d_load)
area.addDock(d_constr, "right", d_controls)

area1.addDock(d_chnl)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# Define the main plot widget

w1 = pg.GraphicsLayoutWidget()
w1.setBackground(background="k")
d_plots.addWidget(w1)
r1 = w1.addPlot(row=0, col=1)  # For data and fit, I(V)
r1_legend = r1.addLegend()
r1_legend.setParentItem(r1)
r1_legend.anchor((0, 1), (0.15, 0.9))

r2 = w1.addPlot(row=0, col=0)  # For V(t) plot
# w1.removeItem(r2)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# Load shot data controls
gui_shot_number = QtGui.QSpinBox()
gui_setup_spinbox(gui_shot_number, 142270, 1, 599999)
gui_shot_label = QtGui.QLabel()
gui_shot_label.setText("shot #")
gui_param_bw = QtGui.QTextBrowser()
gui_ch_label = QtGui.QLabel()
gui_ch_label.setText("Enter channels")
gui_save_btn = QtGui.QPushButton()
gui_save_btn.setText("save fit")
gui_ch = QtGui.QLineEdit()
gui_ch.setText("10")
gui_t0_label = QtGui.QLabel()
gui_t0_label.setText("start, s")
gui_t1_label = QtGui.QLabel()
gui_t1_label.setText("end, s")
gui_t0 = QtGui.QLineEdit()
gui_t0.setText("4.0")
gui_t1 = QtGui.QLineEdit()
gui_t1.setText("4.1")
gui_read_btn = QtGui.QPushButton()
gui_read_btn.setText("read data")
gui_clear_btn = QtGui.QPushButton()
gui_clear_btn.setText("clear all")

gui_move_av = QtGui.QSpinBox()
gui_setup_spinbox(gui_move_av, 5, 1, 50)
gui_move_av.setSuffix(" (N slopes)")
gui_move_av_btn = QtGui.QPushButton()
gui_move_av_btn.setText("average")
# == == == == == == == == == == == == == == == == == == == == == == == == ==
w2 = pg.LayoutWidget()
w2.addWidget(gui_shot_number, 0, 0)
w2.addWidget(gui_shot_label, 0, 1)
w2.addWidget(gui_ch_label, 1, 0)
w2.addWidget(gui_save_btn, 1, 1)
w2.addWidget(gui_ch, 2, 0, 1, 2)
w2.addWidget(gui_t0_label, 3, 0)
w2.addWidget(gui_t1_label, 3, 1)
w2.addWidget(gui_t0, 3, 0)
w2.addWidget(gui_t1, 3, 1)
w2.addWidget(gui_read_btn, 4, 0)
w2.addWidget(gui_clear_btn, 4, 1)

w2.addWidget(gui_move_av, 5, 0)
w2.addWidget(gui_move_av_btn, 5, 1)

w2.addWidget(gui_param_bw, 50, 0, 1, 2)

d_load.addWidget(w2)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# Fit controls
gui_fit_bw = QtGui.QTextBrowser()  # ; gui_fit_bw.setFontPointSize(20)
# gui_next_btn = QtGui.QPushButton(); gui_next_btn.setText('>>>')
# gui_prev_btn = QtGui.QPushButton(); gui_prev_btn.setText('<<<')
gui_vmax_label = QtGui.QLabel()
gui_vmax_label.setText("vmax")
gui_vmax = QtGui.QDoubleSpinBox()
gui_setup_spinbox(gui_vmax, 25.2, 1, 100)
gui_vmax.setSuffix(" V (vmax)")
gui_start_btn = QtGui.QPushButton()
gui_start_btn.setText("start")
# gui_start_btn.setMaximumSize(QtCore.QSize(100, 30))
gui_manual_btn = QtGui.QPushButton()
gui_manual_btn.setText("manual")
gui_i_rampup = QtGui.QSpinBox()
gui_setup_spinbox(gui_i_rampup, 1, 1, 10)
gui_t_label = QtGui.QLabel()
gui_t_label.setText("<t>")
gui_timer_dur = QtGui.QSpinBox()
gui_setup_spinbox(gui_timer_dur, 50, 1, 5000)
gui_timer_dur.setSuffix(" ms (sleep)")
gui_stop_btn = QtGui.QPushButton()
gui_stop_btn.setText("stop")
gui_restart_btn = QtGui.QPushButton()
gui_restart_btn.setText("reset vmax")
gui_mass = QtGui.QSpinBox()
gui_setup_spinbox(gui_mass, 2, 1, 2)
gui_mass.setSuffix(" Da")
gui_dv = QtGui.QSpinBox()
gui_setup_spinbox(gui_dv, 25, 1, 100)
gui_dv.setSuffix(" V (dV)")
gui_chisq = QtGui.QLineEdit()
gui_chisq.setText("5e-2")
gui_chisq_label = QtGui.QLabel()
gui_chisq_label.setText("chisq filter")
# == == == == == == == == == == == == == == == == == == == == == == == == ==
w3 = pg.LayoutWidget()
# w3.addWidget(gui_next_btn,0,0)
# w3.addWidget(gui_prev_btn,0,1)
# w3.addWidget(gui_vmax_label,1,1)

w3.addWidget(gui_restart_btn, 1, 0)
w3.addWidget(gui_t_label, 1, 1)

w3.addWidget(gui_vmax, 2, 0)
w3.addWidget(gui_i_rampup, 2, 1)

w3.addWidget(gui_start_btn, 3, 0)
w3.addWidget(gui_manual_btn, 3, 1)


w3.addWidget(gui_stop_btn, 4, 0)
w3.addWidget(gui_dv, 4, 1)

w3.addWidget(gui_timer_dur, 5, 0)
w3.addWidget(gui_mass, 5, 1)

w3.addWidget(gui_chisq, 6, 0)
w3.addWidget(gui_chisq_label, 6, 1)

w3.addWidget(gui_fit_bw, 8, 0, 1, 2)

d_controls.addWidget(w3)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# Fit summary plots
w4 = pg.GraphicsLayoutWidget()
w4.setBackground(background="k")
d_results.addWidget(w4)
r3 = w4.addPlot(row=0, col=0)  # ne(<t>) plot

r4 = w4.addPlot(row=1, col=0)  # Te(<t>) plot
r5 = w4.addPlot(row=2, col=0)  # Vs(<t>) plot
r4.setXLink(r3)  # link x axis (time) for ne and Te
r5.setXLink(r3)

# r3.setYRange(6e17,2e19,0)
r4.setYRange(0, 30, 0)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# Fit constrains controls
gui_fit_param_bw = QtGui.QTextBrowser()  # ; gui_fit_bw.setFontPointSize(20)
gui_Te_max = QtGui.QDoubleSpinBox()
gui_setup_spinbox(gui_Te_max, 30, 1, 100)
gui_Te_max.setSuffix(" max Te (eV)")
gui_Te_min = QtGui.QDoubleSpinBox()
gui_setup_spinbox(gui_Te_min, 4, 1, 100)
gui_Te_min.setSuffix(" min Te (eV)")
gui_Te_init = QtGui.QDoubleSpinBox()
gui_setup_spinbox(gui_Te_init, 10, 1, 100)
gui_Te_init.setSuffix(" init Te (eV)")

iis_unit = 1e-3
gui_iis_min = QtGui.QDoubleSpinBox()
gui_setup_spinbox(gui_iis_min, 0.5, 0.01, 1e2)
gui_iis_min.setSuffix(" min Iis (mA)")
gui_iis_max = QtGui.QDoubleSpinBox()
gui_setup_spinbox(gui_iis_max, 2e2, 0.1, 1e3)
gui_iis_max.setSuffix(" max Iis (mA)")
gui_iis_init = QtGui.QDoubleSpinBox()
gui_setup_spinbox(gui_iis_init, 1e1, 0.1, 1e3)
gui_iis_init.setSuffix(" init Iis (mA)")
gui_fit_method = QtGui.QComboBox()
fit_method_names = [
    "Nelder-Mead",
    "Levenberg-Marquardt",
    "L-BFGS-B",
    "Powell",
    "Conjugate Gradient",
]
fit_method_keywords = ["nelder", "leastsq", "lbfgsb", "powell", "cg"]
fit_method_dict = {i: j for i, j in zip(fit_method_names, fit_method_keywords)}
gui_fit_method.addItems(fit_method_names)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
w5 = pg.LayoutWidget()
w5.addWidget(gui_fit_method, 1, 0)

w5.addWidget(gui_Te_max, 2, 0)
w5.addWidget(gui_Te_min, 3, 0)
w5.addWidget(gui_Te_init, 4, 0)
w5.addWidget(gui_iis_min, 5, 0)
w5.addWidget(gui_iis_max, 6, 0)
w5.addWidget(gui_iis_init, 7, 0)


w5.addWidget(gui_fit_param_bw, 0, 0, 1, 2)
d_constr.addWidget(w5)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# stdout output browser
gui_cmd_bw = QtGui.QTextBrowser()  # ; gui_fit_bw.setFontPointSize(20)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
w6 = pg.LayoutWidget()

w6.addWidget(gui_cmd_bw, 0, 0, 1, 2)

d_cmd.addWidget(w6)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# Langmuir probe channels selection
gui_lpc_update = QtGui.QPushButton("update")
gui_lpc_all = QtGui.QCheckBox()
gui_lpc_all.setText("select all")
shot = gui_shot_number.value()
lp_info = lp.ch_info(shot)
lp_names = lp_info["ip_ch"].keys()
lp_names = sorted(["{:02d}{}".format(int(i[:-1]), i[-1]) for i in lp_names])
lp_names = ["{}{}".format(int(i[:-1]), i[-1]) for i in lp_names]
# all_ch = [ii for nm in lp_names for ii in lp_info['ip_ch'][nm]]
gui_lpcs = {ii: QtGui.QCheckBox() for ii in lp_names}
[gui_lpcs[ii].setText(ii) for ii in lp_names]

gui_lpc_dict = {
    chnm: [QtGui.QCheckBox() for ii in lp_info["ip_ch"][chnm]] for chnm in lp_names
}
ch_lbl = {chnm: [str(ii) for ii in lp_info["ip_ch"][chnm]] for chnm in lp_names}
[
    [ii.setText(jj) for ii, jj in zip(gui_lpc_dict[chnm], ch_lbl[chnm])]
    for chnm in lp_names
]
# == == == == == == == == == == == == == == == == == == == == == == == == ==
w7 = pg.LayoutWidget()

w7.addWidget(gui_lpc_all, 0, 0, 1, 2)
w7.addWidget(gui_lpc_update, 0, 2, 1, 2)
[w7.addWidget(gui_lpcs[ii], 1, kk) for kk, ii in enumerate(lp_names)]
[
    [w7.addWidget(jj, ii + 2, kk) for ii, jj in enumerate(gui_lpc_dict[chnm])]
    for kk, chnm in enumerate(lp_names)
]

d_chnl.addWidget(w7)

# gui_lpcs['2R'].setChecked(True)
def select_tile_chnls(tile_name):
    """select or deselect all channels for the tile"""
    [ii.setChecked(gui_lpcs[tile_name].isChecked()) for ii in gui_lpc_dict[tile_name]]


def select_all_chnls():
    """select or deselect all"""
    status = gui_lpc_all.isChecked()
    [
        [ii.setChecked(status) for ii, jj in zip(gui_lpc_dict[chnm], ch_lbl[chnm])]
        for chnm in lp_names
    ]
    [gui_lpcs[ii].setChecked(status) for ii in lp_names]


def update_chnl_list():
    """Update channel list"""
    # str(gui_ch.text())
    # for i in lp_names:
    #    for chnl in gui_lpc_dict[i]:
    #        if chnl.isChecked():
    #            print(chnl.text())

    chnls = [
        str(chnl.text())
        for i in lp_names
        for chnl in gui_lpc_dict[i]
        if chnl.isChecked()
    ]
    txt = ", ".join(str(i) for i in chnls)
    gui_ch.setText(txt)


for ii in lp_names:
    gui_lpcs[ii].stateChanged.connect(functools.partial(select_tile_chnls, ii))

gui_lpc_all.stateChanged.connect(select_all_chnls)
gui_lpc_update.clicked.connect(update_chnl_list)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# Disable unused buttons
def enable_btns(btns, enable=True):
    """Disable fit controls"""
    [btn.setEnabled(enable) for btn in btns]


fit_btns = [
    gui_start_btn,
    gui_manual_btn,
    gui_move_av_btn,
    gui_save_btn,
    gui_restart_btn,
    # gui_next_btn,gui_prev_btn,
]
read_btns = [gui_read_btn, gui_clear_btn]
enable_btns(fit_btns, False)
gui_stop_btn.setEnabled(False)
gui_move_av_btn.setEnabled(False)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# End of GUI description
# == == == == == == == == == == == == == == == == == == == == == == == == ==

# == == == == == == == == == == == == == == == == == == == == == == == == ==
""" START  """  # of MAIN FUNCTIONALITY
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# Global variables
vres = None
cur = None
n_rampup = None
t_rampup = None
channels = None
result_dict = None
vmax_array = None
dv_array = None
data_processed = None


def read_channels():
    """read channel list from text box"""
    global channels
    try:
        chnls = str(gui_ch.text())
        chnls = chnls.strip().split(",")
        channels = [int(i) for i in chnls]
    except Exception as err:
        gui_param_bw.append(
            "{}: {}".format(
                time.strftime("%X"),
                err,
            )
        )


def on_read_data():
    """When 'read data' button pressed,
    load data and show some visual info (first ramp-up, or full range)
    """
    if not result_dict == None:
        reply = QtGui.QMessageBox.question(
            win,
            "Message",
            "Read new data? It will clear all plots and data.",
            QtGui.QMessageBox.Yes,
            QtGui.QMessageBox.No,
        )

        if reply == QtGui.QMessageBox.Yes:
            pass
            # print('Saving the fit')
        else:
            # print('Cancel saving')
            return

    clear_all()

    enable_btns(read_btns, False)

    shot = gui_shot_number.value()
    tmin = float(gui_t0.text())
    tmax = float(gui_t1.text())

    read_channels()
    channel = channels[0]

    try:
        write_settings()
    except:
        pass

    arg = {"shot": shot, "channel": channel, "tmin": tmin, "tmax": tmax}

    # pas shot parameters to RetriveData class through process_read
    process_read(arg)
    return


def process_read(arg):
    """Send parameters to RetriveData class to read data"""
    global queue, thread
    queue = que.Queue()
    thread = RetriveData(queue, handle_data)
    thread.start()
    queue.put(arg)
    queue.put(None)


def handle_data(result):
    """Process data read in RetriveData class"""
    global vres, cur, n_rampup, t_rampup, channels
    global result_dict, t_split, vmax_array, dv_array, data_processed

    # out = result.val
    # data_dict = {'vres':vres,'cur':cur}
    data_dict = result.val
    if data_dict is None:
        gui_param_bw.append(
            "{}: {}".format(
                time.strftime("%X"),
                '<font color = "red">Failed to read data. Try to connect to LHD network.</font>',
            )
        )
        enable_btns(read_btns, True)
        enable_btns(fit_btns, True)
        return
    vres = data_dict["vres"]
    cur = data_dict["cur"]

    result.clearClass()

    n_rampup = len(vres["mask"])
    t_rampup = [np.mean(vres["t"][mask]) for mask in vres["mask"]]
    t_split = [vres["t"][mask] for mask in vres["mask"]]
    gui_i_rampup.setMaximum(n_rampup - 1)
    vmax = gui_vmax.value()
    dv = gui_dv.value()

    vmax_array = np.ones(n_rampup) * vmax
    dv_array = np.ones(n_rampup) * dv
    #
    win.statusBar().showMessage("Data loaded")
    # Averaging
    data_processed = {"v": [], "i": [], "t": []}
    average_data()

    gui_i_rampup.setValue(0)
    # mask = vres['mask'][1]
    # out = lp.step_fit_lp(vres,cur,mask,vmax=vmax,dv = dv,dry_run = True)
    # plot_fit(out)
    plot_data()
    gui_param_bw.append(
        "{}: {}".format(
            time.strftime("%X"),
            "Finished reading data",
        )
    )
    # enable controls
    enable_btns(read_btns, True)
    enable_btns(fit_btns, True)
    gui_i_rampup.valueChanged.connect(plot_fit_and_data)


def reshape_result():
    """Reshape result when moving average window changed"""
    global result_dict
    var_names = ["t", "ne", "te", "chisq", "iis", "my_chi", "vs"]
    nan_arr = [np.empty(n_rampup) for i in var_names]
    for i, _ in enumerate(var_names):
        nan_arr[i][:] = np.nan

    result_dict = {var_name: arr for var_name, arr in zip(var_names, nan_arr)}
    result_dict["fit_result"] = {}


def average_data():
    """average data for several ramp-ups"""
    global vres, cur, data_processed, vmax_array, dv_array
    global n_averaged, result_dict

    clear_plot()

    data_processed = {"v": [], "i": [], "t": []}

    di = gui_move_av.value()

    n_rampup = len(vres["mask"]) - 2  # -2 to exclude first and last rampup
    n_averaged = int(n_rampup / di)
    i_start = 1  # start from 1st, skip 0 rampup

    vmax_array = np.ones(n_rampup) * gui_vmax.value()
    dv_array = np.ones(n_rampup) * gui_dv.value()
    # Define result dictionary
    reshape_result()
    for i in range(n_averaged):
        # print(i)
        R = np.arange(i_start + i * di, i_start + di * (i + 1))
        masks = [vres["mask"][j] for j in R]
        max_n = np.min([(len(kk)) for kk in masks])
        masks = [kk[:max_n] for kk in masks]

        vs = np.array([vres["v"][mask] for mask in masks])
        iis = np.array([cur[mask] for mask in masks])
        ts = np.array([vres["t"][mask] for mask in masks])
        # Some problem in the shape of the vs components. Wrong mask?
        # Work-around: try, except - get size of all, get min, artifitially reduce to min
        # print(' '.join(str(pp) for pp in [np.shape(vs[kk])[0] for kk in range(di)] ) )

        data_processed["v"].append(np.sum(vs, 0) / len(vs))
        data_processed["i"].append(np.sum(iis, 0) / len(iis))
        data_processed["t"].append(ts.mean())

    gui_setup_spinbox(gui_i_rampup, 0, 0, n_averaged - 1)


# TEXT formatting for HTML output
def html_float(f, prec=2):
    """TEXT formatting for HTML output"""
    float_str = "{0:.{1}e}".format(f, prec)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return "{0}\u00d710<sup>{1}</sup>".format(base, int(exponent))
    else:
        return float_str


def html_color(s, color="red"):
    """change text color for html output"""
    return '<font color = "{}">{}</font>'.format(color, s)


def html_snc(text, color="red", size=6):
    """change text color for html output"""
    txt = '<font color = "{c}" size="{s}">{t}</font>'.format(c=color, t=text, s=size)
    return txt


# END of TEXT formatting for HTML output


def update_fit_result(out):
    """Update fit result data in the result_dict"""
    global data_processed
    i = gui_i_rampup.value()
    t_av = data_processed["t"][i]
    res = out["result"]

    if out["fit_success"]:
        te = res.values["te"]
        iis = res.values["iis"]
        mass = gui_mass.value()
        vs = res.values["vs"]
        ne = (
            (iis / elementary_charge)
            / ((1.0 / 1000.0) ** 2 * np.pi * 2)
            / (0.61 * (2 * elementary_charge * te / mass / m_p) ** 0.5)
        )

        result_dict["t"][i] = t_av
        result_dict["te"][i] = te
        result_dict["ne"][i] = ne
        result_dict["vs"][i] = vs
        result_dict["iis"][i] = iis
        result_dict["chisq"][i] = res.chisqr
        result_dict["my_chi"][i] = out["my_chi"]


def update_fit_browser():
    """when gui_i_rampup changed, update info displayed for
    the new i_rampup (for a new I-V characteristic for a different time)
    """
    global result_dict
    i = gui_i_rampup.value()

    try:
        t_av = result_dict["t"][i]
        te = result_dict["te"][i]
        ne = result_dict["ne"][i]
        vs = result_dict["vs"][i]
        iis = result_dict["iis"][i]
        chisq = result_dict["chisq"][i]
        my_chi = result_dict["my_chi"][i]

        gui_fit_bw.setText(html_snc(time.strftime("%X"), "black"))
        gui_fit_bw.append(html_snc("&lt;t&gt; = {:.3f}s".format(t_av), "black"))
        txt = "n<sub>e</sub> = {}m<sup>-3</sup>".format(html_float(ne))
        gui_fit_bw.append(html_snc(txt))
        txt = "T<sub>e</sub> = {:.2f}eV".format(te)
        gui_fit_bw.append(html_snc(txt, "blue"))
        txt = "Iis = {:.2e}A".format(iis)
        gui_fit_bw.append(html_snc(txt, "blue"))
        txt = "Vs = {:.2e}V".format(vs)
        gui_fit_bw.append(html_snc(txt, "blue"))
        txt = "chisq = {:.2e}".format(chisq)
        gui_fit_bw.append(html_snc(txt, "green"))

        # txt = 'my_chi = {:.2e}'.format(my_chi)
        # gui_fit_bw.append(html_snc(txt,'black'))

    except IndexError:
        gui_fit_bw.setText(time.strftime("%X"))
        gui_fit_bw.append("No fit data yet")


def plot_fit(out, **kws):
    """Plot fit result"""
    global t_rampup, result_dict, t_split, vres

    if kws.get("clear", False):
        clear_plot()

    i = gui_i_rampup.value()

    v_even = out["v_even"]
    i_even = out["i_even"]

    # == == == == == == == == == == == == == == == =
    # PLOT fit result
    pen = pg.mkPen(
        "y",
        width=3,
    )
    if out["fit_success"]:
        r1.plot(v_even, i_even, pen=pen, name="&nbsp;&nbsp;fit")

    update_fit_browser()


def plot_data(**kws):
    """Plot data for the current time frame
    Update the fit summary plot
    Update the current position point on fit summary plots
    """
    global t_rampup, result_dict, t_split, vres, cur
    global data_processed, chisq_mask

    if kws.get("clear", True):
        clear_plot()

    i = gui_i_rampup.value()
    vmax = gui_vmax.value()
    dv = gui_dv.value()

    # t_step_raw = vres['t'][mask]
    # mask = vres['mask'][i]
    # v_step_raw = vres['v'][mask]
    # i_step_raw = cur[mask]
    # t_step_raw = data_processed['t']
    v_step_raw = data_processed["v"][i]
    i_step_raw = data_processed["i"][i]

    out = lp.step_fit_lp(v_step_raw, i_step_raw, vmax=vmax, dv=dv, dry_run=True)

    # t_av = t_rampup[gui_i_rampup.value()]
    v_f = out["v_f"]
    i_f = out["i_f"]

    # == == == == == == == == == == == == == == == =
    # PLOT Data
    pen = (255, 255, 255, 200)
    r1.plot(
        v_f,
        i_f,
        pen=None,
        symbol="o",
        name="&nbsp;&nbsp;&nbsp;&nbsp;data",
        symbolSize=3,
        symbolPen=pen,
        symbolBrush=pen,
    )
    pen = pg.mkPen(
        "y",
        width=3,
    )

    # == == == == == == == == == == == == == == == =
    # PLOT Voltage vs Time to check the noise
    # vv = vres['v'][mask]
    # tt = t_split[i]
    pen = (255, 255, 255, 230)
    r2.plot(
        v_step_raw,
        pen=None,
        name="&nbsp;&nbsp;V",
        symbol="o",
        symbolSize=2,
        symbolPen=pen,
        symbolBrush=pen,
    )
    pen = (255, 72, 0, 255)
    r2.plot(
        v_step_raw[out["mask_vnoise"]],
        pen=None,
        name="&nbsp;&nbsp;V fil",
        symbol="o",
        symbolSize=2,
        symbolPen=pen,
        symbolBrush=pen,
    )

    # == == == == == == == == == == == == == == == =
    # PLOT FIT SUMMARY
    # mask invalid values (np.NaN or np.Inf)
    t_mask = [not mm for mm in np.ma.masked_invalid(result_dict["t"]).mask]
    ne_mask = [not mm for mm in np.ma.masked_invalid(result_dict["ne"]).mask]
    te_mask = [not mm for mm in np.ma.masked_invalid(result_dict["te"]).mask]
    vs_mask = [not mm for mm in np.ma.masked_invalid(result_dict["vs"]).mask]
    t_ar = result_dict["t"][t_mask]
    ne_ar = result_dict["ne"][ne_mask]
    te_ar = result_dict["te"][te_mask]
    vs_ar = result_dict["vs"][vs_mask]

    chisq_ar = result_dict["chisq"][t_mask]
    # my_chisq_ar = result_dict['my_chi'][t_mask]

    try:
        chisq_max = float(gui_chisq.text())
        chisq_mask = np.where(chisq_ar < chisq_max)
    except:
        chisq_mask = []
        pass

    if len(t_ar):
        pen = (255, 255, 255, 200)
        r3.plot(
            t_ar,
            ne_ar,
            pen=None,
            name="&nbsp;&nbsp;ne",
            symbol="o",
            symbolSize=4,
            symbolBrush=pen,
            symbolPen=pen,
        )

        r4.plot(
            t_ar,
            te_ar,
            pen=None,
            name="&nbsp;&nbsp;Te",
            symbol="o",
            symbolSize=4,
            symbolBrush=pen,
            symbolPen=pen,
        )
        r5.plot(
            t_ar,
            vs_ar,
            pen=None,
            name="&nbsp;&nbsp;Vs",
            symbol="o",
            symbolSize=4,
            symbolBrush=pen,
            symbolPen=pen,
        )
        # plot chisq filter
        pen = (242, 91, 36, 220)
        r3.plot(
            t_ar[chisq_mask],
            ne_ar[chisq_mask],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pen,
            symbolPen=pen,
        )

        r4.plot(
            t_ar[chisq_mask],
            te_ar[chisq_mask],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pen,
            symbolPen=pen,
        )
        # plot current point
        t_now = [result_dict["t"][i]]
        ne_now = [result_dict["ne"][i]]
        te_now = [result_dict["te"][i]]
        vs_now = [result_dict["vs"][i]]
        # print('plot_data', i,t_now,ne_now,te_now)
        pen = (255, 255, 0, 255)
        if len(t_now):
            r3.plot(
                t_now, ne_now, pen=None, symbolPen=None, symbol="o", symbolBrush=pen
            )
            r4.plot(t_now, te_now, pen=None, symbolPen=pen, symbol="o", symbolBrush=pen)
            r5.plot(t_now, vs_now, pen=None, symbolPen=pen, symbol="o", symbolBrush=pen)

        try:
            r3.setLogMode(y=True)
        except:
            pass


def plot_fit_and_data():
    """Plot data and fit if avaliable"""
    global result_dict, vmax_array, dv_array

    i = gui_i_rampup.value()
    gui_vmax.setValue(vmax_array[i])
    gui_dv.setValue(dv_array[i])

    plot_data()
    try:
        out = result_dict["fit_result"][i]
        plot_fit(out, clear=False)
    except KeyError:
        pass


def fit_manual(**kws):
    """Main working function. Fits the data for one time frame.
    Updates the result_dict and the data and fit summary plots
    Updates the fit summary text browser
    """
    global vres, cur, n_rampup, t_rampup, channels, result_dict
    global vmax_array, dv_array
    global iis_unit
    channel = channels[0]

    i = gui_i_rampup.value()
    vmax = gui_vmax.value()
    dv = gui_dv.value()
    vmax_array[i] = vmax
    dv_array[i] = dv
    # mask = vres['mask'][i]
    # t_av = np.mean(vres['t'][mask])

    # 1. get rampup values for rampup number i
    # t_step_raw = vres['t'][mask]
    # v_step_raw = vres['v'][mask]
    # i_step_raw = cur[mask]

    # t_step_raw = data_processed['t']
    v_step_raw = data_processed["v"][i]
    i_step_raw = data_processed["i"][i]

    doloop = kws.get("doloop", False)

    fit_constrains = {
        "te_max": gui_Te_max.value(),
        "te_min": gui_Te_min.value(),
        "te_init": gui_Te_init.value(),
        "iis_min": gui_iis_min.value() * iis_unit,
        "iis_max": gui_iis_max.value() * iis_unit,
        "iis_init": gui_iis_init.value() * iis_unit,
        "method": fit_method_dict[str(gui_fit_method.currentText())],
    }

    if not doloop:
        out = lp.step_fit_lp(
            v_step_raw, i_step_raw, vmax=vmax, dv=gui_dv.value(), **fit_constrains
        )
        if out["fit_success"]:
            update_fit_result(out)
            result_dict["fit_result"][i] = out
            plot_fit_and_data()
        else:
            plot_data()
            gui_fit_bw.setText(
                "{}: {}".format(
                    time.strftime("%X"),
                    '<font color = "red">Fit failed, try different parameters</font>',
                )
            )
    else:
        # loop to select best fit
        chis = []
        vmaxs = []
        outs = []

        for vmax in np.linspace(20, 30, 20):
            out = lp.step_fit_lp(
                v_step_raw, i_step_raw, vmax=vmax, dv=gui_dv.value(), **fit_constrains
            )
            if out["fit_success"]:
                result = out["result"]
                chis.append(result.chisqr)
                vmaxs.append(vmax)
                outs.append(out)

        if len(outs):
            vmax = vmaxs[chis.index(min(chis))]
            gui_vmax.setValue(vmax)
            vmax_array[i] = vmax
            ind = chis.index(min(chis))
            out = outs[ind]

            update_fit_result(out)
            result_dict["fit_result"][i] = out
            plot_fit_and_data()

        else:
            plot_data()
            gui_fit_bw.setText(
                "{}: {}".format(
                    time.strftime("%X"),
                    '<font color = "red">Fit failed, skipping this point</font>',
                )
            )


def fit_loop():
    """Loop through the ramp-ups and try to fit LPA data
    automatically with autoadjustment of the vmax value.
    Initiates the Qt timer, on the timeout this timer calls fit_automatic() function, which
    changes the i_rampup and calls fit_manual()
    The loop starts from i_rampup = 1
    i_rampup = 0 and last one are always ignored, to avoid chekcing if I-V characteristic
    is complete or cropped.
    """
    global vres, cur, n_rampup, t_rampup, timer

    clear_plot()

    enable_btns(fit_btns, False)
    enable_btns(read_btns, False)

    gui_stop_btn.setEnabled(True)
    gui_i_rampup.valueChanged.disconnect(plot_fit_and_data)

    gui_i_rampup.setValue(0)

    timer = QtCore.QTimer()
    timer.timeout.connect(fit_automatic)
    timer.start(gui_timer_dur.value())
    # w1.addItem(r2,1,0)


def fit_automatic():
    """Fit the current i_rampup frame
    Advance i_rampup by 1.
    If this step is last - stop timer and finish automatic fitting
    """
    global timer, vmax_array, dv_array
    i = gui_i_rampup.value()

    gui_vmax.setValue(vmax_array[i])
    gui_dv.setValue(dv_array[i])
    fit_manual(doloop=False)

    if i == gui_i_rampup.maximum():
        gui_param_bw.append(
            "{}: {}".format(
                time.strftime("%X"),
                '<font color = "green">Automatic fit completed</font>',
            )
        )
        stop_timer()
    else:
        gui_i_rampup.setValue(i + 1)


def stop_timer():
    """Stop timer, quit the fitting loop"""
    global timer

    timer.stop()
    gui_stop_btn.setEnabled(False)

    enable_btns(fit_btns, True)
    enable_btns(read_btns, True)
    gui_i_rampup.valueChanged.connect(plot_fit_and_data)


def fit_next():
    """Changes i_rampup and calls fit_manual()"""
    global vmax_array

    gui_i_rampup.valueChanged.disconnect(plot_fit_and_data)

    gui_i_rampup.setValue(gui_i_rampup.value() + 1)
    gui_vmax.setValue(vmax_array[gui_i_rampup.value()])
    gui_dv.setValue(dv_array[gui_i_rampup.value()])
    fit_manual()

    gui_i_rampup.valueChanged.connect(plot_fit_and_data)


def fit_prev():
    """Changes i_rampup and calls fit_manual()"""
    global vmax_array

    gui_i_rampup.valueChanged.disconnect(plot_fit_and_data)

    gui_i_rampup.setValue(gui_i_rampup.value() - 1)
    gui_vmax.setValue(vmax_array[gui_i_rampup.value()])
    gui_dv.setValue(dv_array[gui_i_rampup.value()])
    fit_manual()

    gui_i_rampup.valueChanged.connect(plot_fit_and_data)


def frame_changed():
    """This function is connected to the gui_i_rampup spinbox.
    If the value in that spinbox is changed - this function is called.
    When called, updated the the label gui_t_label to show the averaged time
    of the current frame next to gui_i_rampup spinbox
    """
    global data_processed
    try:
        gui_t_label.setText(
            "{:.3f} s".format(data_processed["t"][gui_i_rampup.value()])
        )
    except TypeError:
        gui_t_label.setText("<t>")


def reset_vmax():
    """Reset vmax_array, set all values in it equal to the
    currently displayed value in gui_vmax spinbox
    """
    global vmax_array, n_averaged
    vmax_array = np.ones(n_averaged) * gui_vmax.value()


def save_data():
    """Save time, ne and Te into a shot_ch_t0-t1.txt
    in the folder, specified in output_path.txt
    """
    from os.path import join
    import os

    global result_dict

    # Better to save shot parameters to global values instead
    # to avoid mistakes and confusion in file names
    shot = gui_shot_number.value()
    tmin = float(gui_t0.text())
    tmax = float(gui_t1.text())
    read_channels()
    channel = channels[0]

    try:
        with open("output_path.txt", "r") as f:
            basepath = f.readline()
    except:
        basepath = "out"
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        gui_param_bw.append(
            "{}: {}".format(
                time.strftime("%X"),
                '<font color = "red">output_path.txt not found. Saving data to working directory</font>',
            )
        )
    shot_pth = join(basepath, "{}".format(shot))
    if not os.path.exists(shot_pth):
        os.makedirs(shot_pth)

    savepath = join(
        shot_pth, "{:d}_ch{:03d}_{:.2f}-{:.2f}.txt".format(shot, channel, tmin, tmax)
    )

    warn_msg = "Overwrite existing data?\n{}".format(savepath)

    if os.path.exists(savepath):
        reply = QtGui.QMessageBox.question(
            win, "Message", warn_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No
        )

        if reply == QtGui.QMessageBox.Yes:
            pass
            # print('Saving the fit')
        else:
            # print('Cancel saving')
            return

    # mask and remove invalid values
    t_mask = [not mm for mm in np.ma.masked_invalid(result_dict["t"]).mask]
    ne_mask = [not mm for mm in np.ma.masked_invalid(result_dict["ne"]).mask]
    te_mask = [not mm for mm in np.ma.masked_invalid(result_dict["te"]).mask]
    vs_mask = [not mm for mm in np.ma.masked_invalid(result_dict["vs"]).mask]
    t_ar = result_dict["t"][t_mask]
    ne_ar = result_dict["ne"][ne_mask]
    te_ar = result_dict["te"][te_mask]
    iis = result_dict["iis"][t_mask]
    vs = result_dict["vs"][vs_mask]
    chisq_ar = result_dict["chisq"][t_mask]
    # print(vs)

    out = np.array([t_ar, te_ar, ne_ar, iis, vs])

    header = "time(s)\tT_e(eV)\tn_e(m^{-3})\tI_s(A)\tV_s(V)"
    fmt = ["%.4f", "%.4f", "%.5e", "%.5e", "%.5e"]
    np.savetxt(savepath, out.T, fmt=fmt, header=header, delimiter="\t")
    gui_param_bw.append(
        "{}: {}".format(
            time.strftime("%X"),
            '<font color = "blue">Fit is saved</font>',
        )
    )

    save_plot([t_ar, ne_ar, te_ar], [shot, channel, tmin, tmax])


def save_plot(data, info):
    """Save summary plot
    data = [t_ar,ne_ar,te_ar]
    info = [shot,channel,tmin,tmax]
    """
    import matplotlib.pylab as plt
    import tools as tls
    from os.path import join
    import os

    global chisq_mask

    t_ar, ne_ar, te_ar = data
    shot, channel, tmin, tmax = info

    tls.font_setup(size=20)

    fig = plt.figure(figsize=(8, 7), facecolor="w")
    gs = plt.GridSpec(2, 1)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])

    ax1.plot(t_ar, ne_ar, "k.")
    ax1.plot(t_ar[chisq_mask], ne_ar[chisq_mask], ".", c="r")

    ax2.plot(t_ar, te_ar, "k.")
    ax2.plot(t_ar[chisq_mask], te_ar[chisq_mask], ".", c="r")

    # ax1.set_ylim(0,np.max(ret['ne'])*1.1)
    # ax2.set_ylim(0,np.max(ret['te'])*1.1)
    [tls.ticks_visual(ax) for ax in [ax1, ax2]]
    [tls.grid_visual(ax) for ax in [ax1, ax2]]
    ax1.set_ylim(5e16, 2e19)
    ax1.set_yscale("log")
    ax2.set_ylim(0, 30)

    ax2.set_xlabel("time, s")
    ax1.set_ylabel("$n_e, m^{-3}$")
    ax2.set_ylabel("$T_e$, eV")

    fig.text(0.2, 0.89, "#{} ch={}".format(shot, channel))

    try:
        with open("output_path.txt", "r") as f:
            basepath = f.readline()
        basepath = join(basepath, "figures")
    except:
        basepath = "out/figures"
        gui_param_bw.append(
            "{}: {}".format(
                time.strftime("%X"),
                '<font color = "red">output_path.txt not found. Saving plot to working directory</font>',
            )
        )
    if not os.path.exists(basepath):
        os.makedirs(basepath)

    savepath = join(
        basepath, "{:d}_ch{:d}_{:.2f}-{:.2f}.png".format(shot, channel, tmin, tmax)
    )
    plt.savefig(
        savepath, dpi=300, bbox_inches="tight", pad_inches=0.1, transparent=False
    )
    gui_param_bw.append(
        "{}: {}".format(
            time.strftime("%X"),
            '<font color = "blue">Plot is saved</font>',
        )
    )


def clear_plot():
    """Clears the pyqtgraph plot widgets, removing all plotted data"""
    try:
        r1.clear()
        r1_legend.items = []
        r2.clear()
        r3.clear()
        r4.clear()
        r5.clear()
    except Exception as e:
        gui_param_bw.append("{}".format(e))

    return


def clear_all():
    """clear all plots. clear all data. clear all browsers."""
    global vres, cur, n_rampup, t_rampup, result_dict, t_split
    global dv_array, vmax_array
    vres = None
    cur = None
    n_rampup = None
    t_rampup = None
    result_dict = None
    data_processed = None
    vmax_array = None
    dv_array = None
    t_split = None

    try:
        gui_i_rampup.valueChanged.disconnect(plot_fit_and_data)
    except:
        pass

    # gui_param_bw.setText('')
    gui_fit_bw.setText("")

    clear_plot()

    enable_btns(fit_btns, False)
    gui_stop_btn.setEnabled(False)


def clear_all_warn():
    """Ask before calling clear_all()"""
    reply = QtGui.QMessageBox.question(
        win,
        "Message",
        "Clear all plots and data?",
        QtGui.QMessageBox.Yes,
        QtGui.QMessageBox.No,
    )

    if reply == QtGui.QMessageBox.Yes:
        clear_all()
    else:
        return


def load_settings():
    """Try to load last shot, channel and start end time from
    the settings.txt in the working folder
    """
    import os
    from os.path import join

    filename = "settings.txt"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f.readlines():
                if "shot number" in line:
                    shot = int(line.split("#")[0].strip())
                if "channel number" in line:
                    channel = int(line.split("#")[0].strip())
                if "start time" in line:
                    t0 = float(line.split("#")[0].strip())
                if "end time" in line:
                    t1 = float(line.split("#")[0].strip())
    else:
        shot = 142270
        channel = 10
        t0 = 4.0
        t1 = 4.1

    gui_shot_number.setValue(shot)
    gui_ch.setText(str(channel))
    gui_t0.setText(str(t0))
    gui_t1.setText(str(t1))


def write_settings():
    """Try to write last shot, channel and start end time from
    the settings.txt in the working folder
    """
    filename = "settings.txt"

    shot = int(gui_shot_number.value())
    channel = str(gui_ch.text())
    t0 = str(gui_t0.text())
    t1 = str(gui_t1.text())

    with open(filename, "w") as f:
        f.write("{} # shot number\n".format(shot))
        f.write("{} # channel number\n".format(channel))
        f.write("{} # start time\n".format(t0))
        f.write("{} # end time\n".format(t1))


# == == == == == == == == == == == == == == == == == == == == == == == == ==
# STDOUT and logging
# on startup, clear the log.txt file or create an empty file in the directory of
# this app if it does not exist.
with open("log.txt", "w") as f:
    f.write("")

# timestamp = '{}> '.format(time.strftime('%X'))


class EmittingStream(QtCore.QObject):

    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


def normalOutputWritten(text):
    """Append text to the QTextEdit."""
    timestamp = "{}> ".format(time.strftime("%X"))

    if len(text) > 1:
        text = timestamp + text

    gui_cmd_bw.insertPlainText(text)

    with open("log.txt", "a") as f:
        f.write(text)


def errorOutputWritten(text):
    """Append text to the QTextEdit."""
    timestamp = "{}> ".format(time.strftime("%X"))
    html_text = text

    if len(text) > 1:
        text = timestamp + text
        if "RuntimeWarning" in text:
            html_text = html_color(text, color="blue") + "<br>"
        else:
            html_text = html_color(text) + "<br>"

    gui_cmd_bw.insertHtml(html_text)

    with open("log.txt", "a") as f:
        f.write(text)


# cmd output to application browser
if CAPTURE_STD:
    sys.stdout = EmittingStream(textWritten=normalOutputWritten)
    sys.stderr = EmittingStream(textWritten=errorOutputWritten)

# END of STDOUT and logging
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# Separate threads to read 'big' data, so the GUI will not freez
class RetriveData(QtCore.QThread):
    taskFinished = QtCore.pyqtSignal(object)

    def __init__(self, queue, callback, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.queue = queue
        self.taskFinished.connect(callback)

    def run(self):
        while True:
            arg = self.queue.get()
            if arg is None:  # None means exit
                return
            self.fun(arg)

    def fun(self, arg):
        # read arg
        # arg = {'shot':shot,'channel':channel,'tmin':tmin,'tmax':tmax}
        shot = arg["shot"]
        channel = arg["channel"]
        tmin = arg["tmin"]
        tmax = arg["tmax"]
        # fetch data
        try:
            vres = lp.get_voltage(shot, tmin, tmax)
            cur = lp.get_current(shot, channel, tmin, tmax)
            # return data back to main app
            data_dict = {"vres": vres, "cur": cur}
            self.taskFinished.emit(ResultObj(data_dict))
        except:
            self.taskFinished.emit(ResultObj(None))


class ResultObj(QtCore.QObject):
    def __init__(self, val):
        self.val = val

    def clearClass(self):
        self.val = None


# == == == == == == == == == == == == == == == == == == == == == == == == ==
# INITIALISE connect GUI to the functions.
gui_read_btn.clicked.connect(on_read_data)
gui_start_btn.clicked.connect(fit_loop)
gui_restart_btn.clicked.connect(reset_vmax)


# gui_next_btn.clicked.connect(fit_next)
# gui_prev_btn.clicked.connect(fit_prev)
gui_manual_btn.clicked.connect(fit_manual)
gui_clear_btn.clicked.connect(clear_all_warn)

gui_save_btn.clicked.connect(save_data)

gui_stop_btn.clicked.connect(stop_timer)

gui_i_rampup.valueChanged.connect(frame_changed)

gui_move_av_btn.clicked.connect(average_data)

try:
    load_settings()
except:
    pass
# End of INITIALISE
# Shortcuts
# shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+F"),win)
# shortcut.activated.connect(fit_manual)

# == == == == == == == == == == == == == == == == == == == == == == == == ==
""" END """  # of MAIN FUNCTIONALITY
# == == == == == == == == == == == == == == == == == == == == == == == == ==
state = area.saveState()


def save_docks():
    global state, area
    state = area.saveState()
    win.statusBar().showMessage("Saved")


def load_docks():
    global state, area
    area.restoreState(state)
    win.statusBar().showMessage("Restored")


save_dock_state = QtGui.QAction("Save Dock State", win)
# save_dock_state.setShortcut('Ctrl+F')
save_dock_state.setStatusTip("Save the Docks layout")
win.connect(save_dock_state, QtCore.SIGNAL("triggered()"), save_docks)

load_dock_state = QtGui.QAction("Load Dock State", win)
# save_dock_state.setShortcut('Ctrl+F')
load_dock_state.setStatusTip("Load the Docks layout")
win.connect(load_dock_state, QtCore.SIGNAL("triggered()"), load_docks)

# == == == == == == == == == == == == == == == == == == == == == == == == ==
fit_displayed = QtGui.QAction("Fit Manually", win)
fit_displayed.setShortcut("Ctrl+F")
fit_displayed.setStatusTip("Fit Currently Displayed Data")
win.connect(fit_displayed, QtCore.SIGNAL("triggered()"), fit_manual)
#
settings_general = QtGui.QAction("General Settings", win)
settings_general.setStatusTip("General Settings are stored in settings.txt")
settings_savepath = QtGui.QAction("Save Directory", win)
settings_savepath.setStatusTip(
    "Absolute Paht of the Saving Directori is in output_path.txt"
)

#
menubar = win.menuBar()
_settings = menubar.addMenu("&Settings")
_settings.addAction(settings_general)
_settings.addAction(settings_savepath)

_fit = menubar.addMenu("&Fit")
_fit.addAction(fit_displayed)

_view = menubar.addMenu("&View")
_view.addAction(save_dock_state)
_view.addAction(load_dock_state)
# == == == == == == == == == == == == == == == == == == == == == == == == ==
# Create the application window and execute
# == == == == == == == == == == == == == == == == == == == == == == == == ==
win.show()

if __name__ == "__main__":
    # START the application main loop. This actually creates the app and
    # Keeps it running
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        # print('Langmuir probe array analysis started')
        QtGui.QApplication.instance().exec_()
