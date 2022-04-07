#!/usr/bin/python3

import sys, os, re
import numpy as np
import scipy.signal as sig


#--------------------------------------------------------------------------
#	Read and check sub_info file.
#
def read_subinfo(para):
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <sub_info filename>")
        exit(0)
    fn = sys.argv[1]
    try:
        f = open(fn, 'r')
    except:
        print(f"!!! read_subinfo: cannot open file: {fn}")
        exit(1)
    lines = f.readlines()
    label = os.path.split(os.path.split(fn)[0])[1]
    f.close()

    Ni, Li, Ri, Hi, gi = -1, -1, -1, -1, -1
    d    = {}
    info = []
    for li, line in enumerate(lines):
        line = line.rstrip()
        if line.find('Index') == 0:
            for i, v in enumerate(line.lower().split()):
                if v == 'num':       Ni=i
                if v == 'lppg_path': Li=i
                if v == 'rppg_path': Ri=i
                if v == 'height':    Hi=i
                if v == 'gender':    gi=i
        elif len(line) > 0 and line[0].isdigit() == True:
            arr = line.split()
            if Ni >= 0 or Li >= 0 or Ri >= 0 or Hi >= 0 or gi >= 0:
                d['label']  = label
                d['i_fn']   = fn
                d['ok']     = 'NULL'
                d['Pid']    = int(arr[Ni])
                d['L_fn']   = arr[Li]
                d['R_fn']   = arr[Ri]
                d['Height'] = float(arr[Hi])
                d['gender'] = int(arr[gi])
                d['lineNo'] = li
                info.append(d.copy())
    return info

def pack_subinfo(info, run, type, idir, odir, rawfn, dFext):
    name  = os.path.split(rawfn)[1]
    label = info['label']
    odir  = os.path.join(odir, label)
    outfn = os.path.join(odir, f"{name}{dFext}.txt")
    rawfn = f"{rawfn}.txt" if os.path.exists(f"{rawfn}.txt") == True else \
               os.path.join(idir, f"{rawfn}.txt")
    dset  = {
        'info':   info,
        'label':  label,
        'name':   name,
        'type':   type,
        'outdir': odir,
        'datafn': rawfn,
        'outfn':  outfn,
        'outfs':  [],
        'run':    run,
        'n_out':  0,
        'sigok':  0,
    }
    return dset

def check_subinfo(para, info):
    idir = os.path.split(info['i_fn'])[0]
    odir = para['outdir']
    dset = []
    type = []
    ok   = {}

    dL = pack_subinfo(info, para['runPPGL'], 'L4', idir, odir,
             info['L_fn'], '_step')
    dR = pack_subinfo(info, para['runPPGR'], 'R4', idir, odir,
             info['R_fn'], '_step')
    for dd in [dL, dR]:
        if dd['run'] != 'yes': continue

        if os.path.exists(dd['outfn']) == False:
            dset.append(dd)
            ok[dd['type']] = 0
        else:
            ok[dd['type']] =-1
        type.append(dd['type'])
    if len(dset) > 0:
        os.makedirs(dset[0]['outdir'], exist_ok=True)
        os.makedirs(para['wdir'], exist_ok=True)
    info['type'] = type.copy()
    info['ok']   = ok.copy()
    return dset

#--------------------------------------------------------------------------
#	Utilities.
#
def search_t_idx(nT, t, t0):
    if (t[0]    >= t0): return 0
    if (t[nT-1] <= t0): return nT-1

    i0 = int(nT/2)
    di = int(nT/4)
    while (t[i0] > t0 or t[i0+1] <= t0) and di > 0:
        i0 = i0-di if (t[i0] > t0) else i0+di
        di = int(di/2)
    if t[i0+1] <= t0:
        while i0 < nT:
            if t[i0] <= t0 and t[i0+1] > t0: break
            i0 = i0+1
    elif t[i0] > t0:
        while i0 >= 0:
            if t[i0] <= t0 and t[i0+1] > t0: break
            i0 = i0-1
    if i0 < nT-1 and i0 >= 0:
        return i0 if t[i0] <= t0 and t[i0+1] > t0 else -1
    else:
        return 0 if i0 < 0 else nT-1

def move_outfiles(data):
    for f in data['outfs']:
        f   = os.path.basename(f)
        fn1 = os.path.join(para['wdir'],   f)
        fn2 = os.path.join(data['outdir'], f)
        os.rename(fn1, fn2)

#--------------------------------------------------------------------------
#	Signal interpolation algorithms.
#
def InterpSimple(x0, f0, x1):
    fg = []
    k  = 0
    n  = len(x0)
    for i in range(len(x1)):
        while k < n-1 and x0[k] < x1[i]: k=k+1
        k = k-1
        r = f0[k] + (f0[k+1]-f0[k]) * (x1[i]-x0[k]) / (x0[k+1]-x0[k])
        fg.append(r)
    return fg

def InterpDataNorm(x0, f0):
    ff0  = np.asarray(f0, dtype=float)
    xx0  = np.asarray(x0, dtype=float)
    fmax = np.abs(ff0).max()
    ff0  = ff0 / fmax

    dx   = xx0[1] - xx0[0]
    xscl = 1
    while dx < 0.1:
        xscl = xscl * 10
        dx   = dx   * 10
    while dx > 1.0:
        xscl = xscl / 10
        dx   = dx   / 10
    x0R = xx0 * xscl
    return fmax, x0R, ff0

def InterpAkima(x0, f0, x1):
    n  = len(x0)
    k  = 0
    fg = []
    ss = np.zeros(n+3, dtype=float)
    f1 = np.zeros(n, dtype=float)
    c0 = np.zeros(n, dtype=float)
    c1 = np.zeros(n, dtype=float)
    c2 = np.zeros(n, dtype=float)
    c3 = np.zeros(n, dtype=float)
    fscl, x0R, f0R = InterpDataNorm(x0, f0)
    for i in range(n-1):
        ss[i+2] = (f0R[i+1] - f0R[i]) / (x0R[i+1] - x0R[i])
    ss[1]   = 2.0*ss[2] - ss[3]
    ss[0]   = 2.0*ss[1] - ss[2]
    ss[n+1] = 2.0*ss[n] - ss[n-1]
    ss[n+2] = 2.0*ss[n+1] - ss[n]

    for i in range(n):
        dn = np.abs(ss[i+3]-ss[i+2]) + np.abs(ss[i]-ss[i+1])
        f1[i] = 0.0 if (dn == 0.0) else \
                (np.abs(ss[i+3]-ss[i+2])*ss[i+1] +
                 np.abs(ss[i] - ss[i+1])*ss[i+2]) / dn
    for i in range(n-1):
        c0[i] = f0R[i]
        c1[i] = f1[i]
        c2[i] = 3.0*(f0R[i+1]-f0R[i]-f1[i]) - (f1[i+1]-f1[i])
        c3[i] = (f1[i+1]-f1[i]) - 2.0*(f0R[i+1]-f0R[i]-f1[i])
    for i in range(len(x1)):
        while k < n-1 and x0[k] < x1[i]: k=k+1
        k = k-1
        if k > n-1: break
        u = (x1[i] - x0[k]) / (x0[k+1] - x0[k])
        r = (c0[k] + c1[k]*u + c2[k]*u*u + c3[k]*u*u*u)*fscl
        fg.append(r)
    return fg

def SignalInterp(para, x0, f0, x1):
    if (para['interp_algo']) == 1:
        return InterpSimple(x0, f0, x1)
    elif (para['interp_algo']) == 2:
        return InterpAkima(x0, f0, x1)
    else:
        print(f"!!! Unknown signal interpolation: {para['interp_algo']}")
        exit(1)

#--------------------------------------------------------------------------
#	Read the source data file (format 3).
#
def read_src(para, dset):
    sigIR0, sigRD0, t0 = [], [], []
    sigIR,  sigRD,  t  = [], [], []
    t_skip = para['T_init_skip']

    try:
        f = open(dset['datafn'], 'r')
    except:
        print(f"!!! read_src: cannot open file: {dset['datafn']}")
        return False
    lines = f.readlines() 
    f.close()

    for line in lines[1:]:
        line = line.rstrip().lstrip()
        arr  = line.split()
        sigIR0.append(-float(arr[0]))
        sigRD0.append(-float(arr[1]))

    nT = len(sigIR0)
    dsmpraw = int(nT/12000 + 0.5)
    if dsmpraw == 0: dsmpraw=1

    dsample = para['dsample']
    T_tot   = para['ttot'] + t_skip
    ts      = -1
    dT      = T_tot / (nT-1)
    for i in range(nT):
        tx = i * dT
        if tx < t_skip: continue
        if ts == -1: ts=tx
        t0.append(tx-ts)
        sigIR.append(sigIR0[i])
        sigRD.append(sigRD0[i])

    nT = int(dsample * T_tot)
    dT = T_tot / (nT-1)
    for i in range(nT-1):
        tx = i * dT
        if tx >= t_skip:
            t.append(tx - t_skip)
    dset['sigIR']   = SignalInterp(para, t0, sigIR, t)
    dset['sigRD']   = SignalInterp(para, t0, sigRD, t)
    dset['nT']      = len(t)
    dset['t']       = t
    dset['dsmpraw'] = dsmpraw

    return True

#--------------------------------------------------------------------------
#	Output data.
#
def out_AC_DC_data(para, data):
    res = data['res']
    fn  = os.path.join(para['wdir'], f"{data['name']}_signal.txt")
    try:
        f = open(fn, 'wt')
    except:
        print(f"!!! out_AC_DC_data: cannot output file: {fn}")
        exit(1)
    print("#", file=f)
    print("#   sig       i0     iN        t0       tN", file=f)
    print("#   --------------------------------------------", file=f)
    if data['sigok'] == True:
        for i, rs in enumerate(res):
            i0, iN, t = rs['i0'], rs['iN'], rs['t']
            print(f"#   {rs['signame']}    {i0:6d} {iN:6d}  {t[i0]:8.3f} {t[iN]:8.3f}", file=f)
    print("#", file=f)
    print("#t       signal_IR          DC             AC         signal_RD          DC             AC", file=f);

    rs1, rs2 = res[0], res[1]
    for i in range(rs['nT']):
        print(f"{rs1['t'][i]:.4f}  {rs1['sig'][i]:13.6E}  {rs1['sDC'][i]:13.6E}  {rs1['sAC'][i]:13.6E}  {rs2['sig'][i]:13.6E}  {rs2['sDC'][i]:13.6E}  {rs2['sAC'][i]:13.6E}", file=f)
    f.close()
    data['outfs'].append(fn)

def out_min_max(para, data):
    res = data['res']
    fn  = os.path.join(para['wdir'], f"{data['name']}_minmax.txt")
    try:
        f = open(fn, 'wt')
    except:
        print(f"!!! out_min_max: cannot output file: {fn}")
        exit(1)
    for rs in res:
        t,    sAC,  pulse = rs['t'],    rs['sAC'],  rs['pulse']
        lmin, lmax, lmax2 = rs['lmin'], rs['lmax'], rs['lmax2']
        print("#Label                   Name  Sig  Step  M/m  index  t(sec)            Amp", file=f)
        for idx in lmin:
            print(f"{data['label']:<12s} {data['name']:13s}  {rs['signame']:3s} {1:4d}  lmin  {idx:5d}  {t[idx]:6.3f}  {sAC[idx]:13.6E}", file=f)
        for idx in lmax:
            print(f"{data['label']:<12s} {data['name']:13s}  {rs['signame']:3s} {1:4d}  lmax  {idx:5d}  {t[idx]:6.3f}  {sAC[idx]:13.6E}", file=f)
        for idx in lmax2:
            print(f"{data['label']:<12s} {data['name']:13s}  {rs['signame']:3s} {1:4d}  lmx2  {idx:5d}  {t[idx]:6.3f}  {sAC[idx]:13.6E}", file=f)
        for p in pulse:
            print(f"{data['label']:<12s} {data['name']:13s}  {rs['signame']:3s} {1:4d}    DN  {p['iDN']:5d}  {t[p['iDN']]:6.3f}  {sAC[p['iDN']]:13.6E}", file=f)
        for p in pulse:
            print(f"{data['label']:<12s} {data['name']:13s}  {rs['signame']:3s} {1:4d}    DP  {p['iDP']:5d}  {t[p['iDP']]:6.3f}  {sAC[p['iDP']]:13.6E}", file=f)
        print("", file=f)
    f.close()
    data['outfs'].append(fn)

    fn = os.path.join(para['wdir'], f"{data['name']}_peaklist.txt")
    try:
        f = open(fn, 'wt')
    except:
        print(f"!!! out_min_max: cannot output file: {fn}")
        exit(1)
    for rs in res:
        t, sAC, pulse = rs['t'], rs['sAC'], rs['pulse']
        print("#Label                   Name  Sig  pulse", file=f)
        for j, p in enumerate(pulse):
            pulse_vally, pulse_peak = p['pulse_vally'], p['pulse_peak']
            print(f"{data['label']:<12s} {data['name']:8s}  {rs['signame']:3s} {j:4d}   ", end='', file=f)
            for pp in pulse_peak:
                print(f"  P  {t[pp]:8.4f}  {sAC[pp]:12.4E}", end='', file=f)
            for pp in pulse_vally:
                print(f"  V  {t[pp]:8.4f}  {sAC[pp]:12.4E}", end='', file=f)
            print('', file=f)
        print('', file=f)
    f.close()
    data['outfs'].append(fn)

def out_dA1_res(para, data):
    label, name, res = data['label'], data['name'], data['res']
    fn = os.path.join(para['wdir'], f"{data['name']}_dA1res.txt")
    try:
        f = open(fn, 'wt')
    except:
        print(f"!!! out_dA1_res: cannot output file: {fn}")
        exit(1)
    for rs in res:
        signame, t, pulse, ave = rs['signame'], rs['t'], rs['pulse'], rs['ave']
        if signame == 'IR':
            print("#                                                      SI001        SI002        SI003        SI004        SI007        SI008        SI009        SI010        SI011        SI012        SI013        SI014        SI015        SI016        SI017        SI005        SI006             SI018             SI019", file=f)
        else:
            print("#                                                      SR001        SR002        SR003        SR004        SR007        SR008        SR009        SR010        SR011        SR012        SR013        SR014        SR015        SR016        SR017        SR005        SR006             SR018             SR019", file=f)
        print("#Label                   Name   Sig  Step  pulse          DC           LM           Lm          Amp          HBT        DN(t)           DN        DP(t)           DP            a           x0            s     plat(t1)     plat(t2)    plat(len)       CresTa       CresTb   amp(dA1)/CresTa   amp(dA1)/CresTb", file=f)
        for k, p in enumerate(pulse):
            print(f"{label:<12s} {name:16s}  {signame:>3s} {1:4d}   {k:4d}  {p['aDC']:12.4E} {p['LM']:12.4E} {p['Lm']:12.4E} {p['Ampval']:12.4E} {p['HBTval']:12.4E} {p['tDNval']:12.4E} {p['DNval']:12.4E} {p['tDPval']:12.4E} {p['DPval']:12.4E} {p['a']:12.4E} {p['x0']:12.4E} {p['s']:12.4E} {p['tip1']:12.4E} {p['tip2']:12.4E} {p['tiplen']:12.4E} {p['crTa']:12.4E} {p['crTb']:12.4E} {p['ACTa']:17.4E} {p['ACTb']:17.4E}", file=f)
        print(f"{label:<12s} {name:16s}  {signame:>3s} {1:4d}    ave  {ave['aveDC']:12.4E} {ave['aLM']:12.4E} {ave['aLm']:12.4E} {ave['aveAmp']:12.4E} {ave['HBT']:12.4E} {ave['tDN']:12.4E} {ave['aDN']:12.4E} {ave['tDP']:12.4E} {ave['aDP']:12.4E} {ave['dA1_a']:12.4E} {ave['dA1_x0']:12.4E} {ave['dA1_s']:12.4E} {ave['dA1_pt1']:12.4E} {ave['dA1_pt2']:12.4E} {ave['dA1_plen']:12.4E} {ave['CresTa']:12.4E} {ave['CresTb']:12.4E} {ave['dA1_ACTa']:17.4E} {ave['dA1_ACTb']:17.4E}", file=f)
        print(f"{label:<12s} {name:16s}  {signame:>3s} {1:4d}    ave  {ave['errDC']:12.4E} {ave['dLM']:12.4E} {ave['dLm']:12.4E} {ave['errAmp']:12.4E} {ave['dHBT']:12.4E} {ave['dtDN']:12.4E} {ave['dDN']:12.4E} {ave['dtDP']:12.4E} {ave['dDP']:12.4E} {ave['dA1_da']:12.4E} {ave['dA1_dx0']:12.4E} {ave['dA1_ds']:12.4E} {ave['dA1_dpt1']:12.4E} {ave['dA1_dpt2']:12.4E} {ave['dA1_dplen']:12.4E} {ave['dCresTa']:12.4E} {ave['dCresTb']:12.4E} {ave['dA1_dACTa']:17.4E} {ave['dA1_dACTb']:17.4E}", file=f)
        print('', file=f)
    f.close()
    data['outfs'].append(fn)

def out_peak_feature(para, data):
    res = data['res']
    fn  = os.path.join(para['wdir'], f"{data['name']}_peak.txt")
    try:
        f = open(fn, 'wt')
    except:
        print(f"!!! out_peak_feature: cannot output file: {fn}")
        exit(1)
    for rs in res:
        pulse, ave = rs['pulse'], rs['ave']
        print('#', file=f)
        print(f"#    {data['label']:s}: {data['name']:s}, {rs['signame']:s}, step {1:d}", file=f)
        print('#', file=f)
        print(f"#    {'Signal_peak':<65s}{'Signal_valley':<65s}{'dSignal_peak':<65s}{'dSignal_valley':<65s}", file=f)
        if rs['signame'] == 'IR':
            print("#    SI020   SI021         SI022         SI023         SI024                                SI025         SI026         SI027          SI028   SI029         SI030         SI031         SI032          SI033   SI034         SI035         SI036         SI037          SI038         SI039         SI040         SI041         SI044         SI046          SI048         SI049         SI050", file=f)
        else:
            print("#    SR020   SR021         SR022         SR023         SI024                                SR025         SR026         SR027          SR028   SR029         SR030         SR031         SR032          SR033   SR034         SR035         SR036         SI037          SR038         SR039         SR040         SR041         SR044         SR046          SR048         SR049         SR050", file=f)
        print("#no  ", end='', file=f)
        for k in range(4):
            print("t0      Amp           curvature     L-slope       R-slope        ", end='', file=f)
        print("PWV1          PWV2          PTT1          PTT2          dT_intvl1     dT_intvl2     N_Peaks       N_Vallys      TailSlope", file=f)
        for k, p in enumerate(pulse):
            print(f"{k:03d}", end='', file=f)
            print(f"{p['res1']['t']:8.4f}{p['res1']['pval']:14E}{p['res1']['curv']:14E}{p['res1']['slopeL']:14E}{p['res1']['slopeR']:14E} ", end='', file=f)
            print(f"{0:8.4f}{0:14E}{p['res2']['curv']:14E}{p['res2']['slopeL']:14E}{p['res2']['slopeR']:14E} ", end='', file=f)
            print(f"{p['res3']['t']:8.4f}{p['res3']['pval']:14E}{p['res3']['curv']:14E}{p['res3']['slopeL']:14E}{p['res3']['slopeR']:14E} ", end='', file=f)
            print(f"{p['res4']['t']:8.4f}{p['res4']['pval']:14E}{p['res4']['curv']:14E}{p['res4']['slopeL']:14E}{p['res4']['slopeR']:14E} ", end='', file=f)
            print(f"{0:14E}{0:14E}{0:14E}{0:14E}{0:14E}{0:14E}{p['npulse_peak']:14E}{p['npulse_vally']:14E}{p['TailSlope']:14E}", file=f)
        print('ave', end='', file=f)
        print(f"{ave['pres1']['t']:8.4f}{ave['pres1']['Amp']:14E}{ave['pres1']['curv']:14E}{ave['pres1']['slopeL']:14E}{ave['pres1']['slopeR']:14E} ", end='', file=f)
        print(f"{0:8.4f}{0:14E}{ave['pres2']['curv']:14E}{ave['pres2']['slopeL']:14E}{ave['pres2']['slopeR']:14E} ", end='', file=f)
        print(f"{ave['pres3']['t']:8.4f}{ave['pres3']['Amp']:14E}{ave['pres3']['curv']:14E}{ave['pres3']['slopeL']:14E}{ave['pres3']['slopeR']:14E} ", end='', file=f)
        print(f"{ave['pres4']['t']:8.4f}{ave['pres4']['Amp']:14E}{ave['pres4']['curv']:14E}{ave['pres4']['slopeL']:14E}{ave['pres4']['slopeR']:14E} ", end='', file=f)
        print(f"{0:14E}{0:14E}{0:14E}{0:14E}{0:14E}{0:14E}{ave['aPpeak']:14E}{ave['aPvally']:14E}{ave['aTailSlope']:14E}", file=f)
        print('err', end='', file=f)
        print(f"{ave['pres1']['dt']:8.4f}{ave['pres1']['dAmp']:14E}{ave['pres1']['dcurv']:14E}{ave['pres1']['dslopeL']:14E}{ave['pres1']['dslopeR']:14E} ", end='', file=f)
        print(f"{0:8.4f}{0:14E}{ave['pres2']['dcurv']:14E}{ave['pres2']['dslopeL']:14E}{ave['pres2']['dslopeR']:14E} ", end='', file=f)
        print(f"{ave['pres3']['dt']:8.4f}{ave['pres3']['dAmp']:14E}{ave['pres3']['dcurv']:14E}{ave['pres3']['dslopeL']:14E}{ave['pres3']['dslopeR']:14E} ", end='', file=f)
        print(f"{ave['pres4']['dt']:8.4f}{ave['pres4']['dAmp']:14E}{ave['pres4']['dcurv']:14E}{ave['pres4']['dslopeL']:14E}{ave['pres4']['dslopeR']:14E} ", end='', file=f)
        print(f"{0:14E}{0:14E}{0:14E}{0:14E}{0:14E}{0:14E}{ave['dPpeak']:14E}{ave['dPvally']:14E}{ave['dTailSlope']:14E}", file=f)
    f.close()
    data['outfs'].append(fn)

def out_PPG_sigFT(para, data):
    nT    = data['nT']
    tstep = (data['t'][nT-1] - data['t'][0]) / (nT-1)
    Fs    = 1.0 / tstep
    dF    = Fs  / nT
    res   = data['res']
    for rs in res:
        sig   = rs['sAC']
        sname = rs['signame']
        PPGF  = np.absolute(np.fft.fft(sig))

        fn = os.path.join(para['wdir'], f"{data['name']}_PPG_sig{sname}FT.txt")
        try:
            f = open(fn, 'wt')
        except:
            print(f"!!! out_PPG_sigFT: cannot output file: {fn}")
            exit(1)
        print("#frequency      power_spectrum", file=f)
        for i in range(1, nT):
            print(f"{dF*i:12.6E}   {PPGF[i]**2:15.8E}", file=f)
            if dF*i > 15.0: break
        f.close()
        data['outfs'].append(fn)

def res_collect(rs):
    val, val0, val1 = [], [], []
    pave         = rs['ave']
    pres1, pres2 = pave['pres1'], pave['pres2']
    pres3, pres4 = pave['pres3'], pave['pres4']

    val0.append(pave['aveDC'])
    val0.append(pave['aLM'])
    val0.append(pave['aLm'])
    val0.append(pave['aveAmp'])
    val1.append(pave['errDC'])
    val1.append(pave['dLM'])
    val1.append(pave['dLm'])
    val1.append(pave['errAmp'])

    val0.append(pave['CresTa'])
    val0.append(pave['CresTb'])
    val0.append(pave['HBT'])
    val1.append(pave['dCresTa'])
    val1.append(pave['dCresTb'])
    val1.append(pave['dHBT'])

    val0.append(pave['tDN'])
    val0.append(pave['aDN'])
    val0.append(pave['tDP'])
    val0.append(pave['aDP'])
    val1.append(pave['dtDN'])
    val1.append(pave['dDN'])
    val1.append(pave['dtDP'])
    val1.append(pave['dDP'])

    val0.append(pave['dA1_a'])
    val0.append(pave['dA1_x0'])
    val0.append(pave['dA1_s'])
    val1.append(pave['dA1_da'])
    val1.append(pave['dA1_dx0'])
    val1.append(pave['dA1_ds'])

    val0.append(pave['dA1_pt1'])
    val0.append(pave['dA1_pt2'])
    val0.append(pave['dA1_plen'])
    val0.append(pave['dA1_ACTa'])
    val0.append(pave['dA1_ACTb'])
    val1.append(pave['dA1_dpt1'])
    val1.append(pave['dA1_dpt2'])
    val1.append(pave['dA1_dplen'])
    val1.append(pave['dA1_dACTa'])
    val1.append(pave['dA1_dACTb'])

    val0.append(pres1['t'])
    val0.append(pres1['Amp'])
    val0.append(pres1['curv'])
    val0.append(pres1['slopeL'])
    val0.append(pres1['slopeR'])
    val1.append(pres1['dt'])
    val1.append(pres1['dAmp'])
    val1.append(pres1['dcurv'])
    val1.append(pres1['dslopeL'])
    val1.append(pres1['dslopeR'])

    val0.append(pres2['curv'])
    val0.append(pres2['slopeL'])
    val0.append(pres2['slopeR'])
    val1.append(pres2['dcurv'])
    val1.append(pres2['dslopeL'])
    val1.append(pres2['dslopeR'])

    val0.append(pres3['t'])
    val0.append(pres3['Amp'])
    val0.append(pres3['curv'])
    val0.append(pres3['slopeL'])
    val0.append(pres3['slopeR'])
    val1.append(pres3['dt'])
    val1.append(pres3['dAmp'])
    val1.append(pres3['dcurv'])
    val1.append(pres3['dslopeL'])
    val1.append(pres3['dslopeR'])

    val0.append(pres4['t'])
    val0.append(pres4['Amp'])
    val0.append(pres4['curv'])
    val0.append(pres4['slopeL'])
    val0.append(pres4['slopeR'])
    val1.append(pres4['dt'])
    val1.append(pres4['dAmp'])
    val1.append(pres4['dcurv'])
    val1.append(pres4['dslopeL'])
    val1.append(pres4['dslopeR'])

    val0.append(0.0)
    val0.append(0.0)
    val0.append(0.0)
    val0.append(0.0)
    val0.append(0.0)
    val0.append(0.0)
    val0.append(0.0)
    val0.append(0.0)
    val0.append(0.0)
    val0.append(0.0)
    val1.append(0.0)
    val1.append(0.0)
    val1.append(0.0)
    val1.append(0.0)
    val1.append(0.0)
    val1.append(0.0)
    val1.append(0.0)
    val1.append(0.0)
    val1.append(0.0)
    val1.append(0.0)

    val0.append(pave['aPpeak'])
    val0.append(pave['aPvally'])
    val0.append(pave['aTailSlope'])
    val1.append(pave['dPpeak'])
    val1.append(pave['dPvally'])
    val1.append(pave['dTailSlope'])
    val.append(val0)
    val.append(val1)
    return val

def out_features(f, val):
    vIR, vRD = val[0], val[1]
    n = len(vIR[0])
    for i in range(n):
        print(f"SI{i+1:03d}A  ", end='', file=f)
        if vIR[0][i] == 'n/a':
            print(f"{'n/a':17s}", file=f)
        else:
            print(f"{vIR[0][i]:17.8E}", file=f)
        print(f"SI{i+1:03d}V  ", end='', file=f)
        if vIR[1][i] == 'n/a':
            print(f"{'n/a':17s}", file=f)
        else:
            print(f"{vIR[1][i]:17.8E}", file=f)

        print(f"SR{i+1:03d}A  ", end='', file=f)
        if vRD[0][i] == 'n/a':
            print(f"{'n/a':17s}", file=f)
        else:
            print(f"{vRD[0][i]:17.8E}", file=f)
        print(f"SI{i+1:03d}V  ", end='', file=f)
        if vRD[1][i] == 'n/a':
            print(f"{'n/a':17s}", file=f)
        else:
            print(f"{vRD[1][i]:17.8E}", file=f)

def out_ave(para, data):
    tseg, res  = para['T_step'], data['res']
    rs, vals   = res[0], []
    t, i0, iN  = rs['t'], rs['i0'], rs['iN']
    for rs in res:
        val = res_collect(rs)
        vals.append(val)

    fn = os.path.join(para['wdir'], f"{data['name']}_step.txt")
    try:
        f = open(fn, 'wt')
    except:
        print(f"!!! out_ave: cannot output file: {fn}")
        exit(1)
    print(f"{'Label':<8s}{data['label']:>17s}", file=f)
    print(f"{'Name':<8s}{data['name']:>17s}", file=f)
    print(f"{'Step':<8s}{1:17d}", file=f)
    print(f"{'Step(t0)':<8s}{t[i0]:17.4f}", file=f)
    print(f"{'Step(tN)':<8s}{t[iN]:17.4f}", file=f)
    print(f"{'StepLen':<8s}{(t[iN]-t[i0])/tseg*100:16.3f}%", file=f)
    out_features(f, vals)
    f.close()
    data['outfs'].append(fn)

#--------------------------------------------------------------------------
#	PPG signal preprocessing
#
def ppg_ac_normalize(sAC):
    ave = np.mean(sAC)
    err = np.std(sAC)
    sAC = (sAC - ave) / err
    return sAC

def get_all_minmax(para, dt, nT, sig, i0, iN):
    """ Bigger Fall Side Detection algorighm """
    lmax, lmax2, lmin = [], [], []
    tx,   px,    dp   = [], [], []
    for i in range(1, nT-1):
        if (sig[i]-sig[i-1])*(sig[i+1]-sig[i]) < 0:
            tx.append(i)
            px.append(sig[i])
    np = len(tx)

    for i in range(np-1):
        r = 0 if px[i] >= px[i+1] else px[i+1]-px[i]
        dp.append(r)
    d2 = dp.copy()
    d2.sort(reverse=True)
    dm = d2[20]

    TH_Ratio = para['TH_Ratio']
    for i in range(len(dp)):
        if dp[i] < dm*TH_Ratio or dp[i] > dm/TH_Ratio: continue
        if sig[tx[i]] < 0.0 and sig[tx[i+1]] > 0.0:
            lmin.append(tx[i])
            lmax.append(tx[i+1])
    j = 0
    for i in range(np):
        if j >= len(lmax):  break
        if tx[i] < lmax[j]: continue
        if tx[i] == lmax[j]:
            k   = i
            dmm = px[i]
        elif j < len(lmin)-1 and tx[i] < lmin[j+1]:
            if dmm < px[i]:
                k   = i
                dmm = px[i]
        else:
            lmax2.append(tx[k])
            j = j+1
    return lmax, lmax2, lmin

def sep_AC_DC(para, data, signame, i0, iN):
    order = 6
    ok    = True
    t     = data['t']
    nT    = data['nT']
    v     = np.asarray(data[f"sig{signame}"], dtype=float)
    F     = int(1.0/((t[nT-1] - t[0]) / (nT-1)) + 0.5)
    res   = { 't': t, 'nT': nT, 'sig': v }

    freq = para['NZ_Hz']
    if freq != 0:
        b, a = sig.butter(order, 2*freq/F)
        npad = 3 * (np.maximum(b.size, a.size) - 1)
        v    = sig.filtfilt(b, a, v, padtype='odd', padlen=npad)
        res['sig'] = v

    freq = para['AC_Hz']
    b, a = sig.butter(order, 2*freq/F)
    npad = 3 * (np.maximum(b.size, a.size) - 1)
    sDC  = sig.filtfilt(b, a, v, padtype='odd', padlen=npad)
    sAC  = v - sDC

    dt = t[1] - t[0]
    nn = int(1.0 / dt)
    j, x0, y0 = i0, [], []
    while j < iN and j < nT:
        n    = nn if j+nn <= nT else nT-j
        lmin = sAC[j:j+n].min()
        lmax = sAC[j:j+n].max()
        x0.append(j)
        y0.append((lmin+lmax)/2.0)
        j = j + n
    for i in range(len(x0)-1):
        dy = (y0[i+1] - y0[i]) / (x0[i+1] - x0[i])
        k  = [ y0[i]+(j-x0[i])*dy for j in range(x0[i], x0[i+1]) ]
        k  = np.asarray(k)
        sAC[x0[i]:x0[i+1]] = sAC[x0[i]:x0[i+1]] - k
        sDC[x0[i]:x0[i+1]] = sDC[x0[i]:x0[i+1]] + k
    sAC[x0[-1]:nT] = sAC[x0[-1]:nT] - y0[-1]
    sDC[x0[-1]:nT] = sDC[x0[-1]:nT] + y0[-1]

    if para['PPG_Norm'] == 1:
        sAC = ppg_ac_normalize(sAC)

    lmax, lmax2, lmin = get_all_minmax(para, dt, nT-30, sAC, i0, iN)
    if len(lmin) <= 1 or len(lmax) <= 1 or len(lmax2) <= 1:
        print(f"!!! {signame}: The signal is too noisy to find any pulse.")
        print(f"!!! {signame}: Please try to tune the following parameters:")
        print(f"!!!         DATA_SAMPLE_R, NZ_HZ, TH_RATIO.")
        print(f"!!! {signame}: ignore this signal.")
        ok = False

    res['signame'] = signame
    res['sDC']     = sDC
    res['sAC']     = sAC
    res['lmax']    = np.asarray(lmax)
    res['lmax2']   = np.asarray(lmax2)
    res['lmin']    = np.asarray(lmin)
    res['nmax']    = len(lmax)
    res['nmax2']   = len(lmax2)
    res['nmin']    = len(lmin)
    res['i0']      = i0
    res['iN']      = iN
    return ok, res

def PPG_preproc(para, data, signame):
    nT = data['nT']
    t  = data['t']
    t0 = para['T_dt_stepb']
    tN = para['T_step']
    i0 = search_t_idx(nT, t, t0)
    iN = search_t_idx(nT, t, tN)
    if i0 == -1 or iN == -1:
        print(f"!!! PPG_preproc: cannot find t-idx: t0=[{t0:f},{tN:f}], idx=[{i0}:{iN}]");
        exit(1)
    ok, res = sep_AC_DC(para, data, signame, i0, iN)
    return res, ok

#--------------------------------------------------------------------------
#	PPG signal feature extraction
#
def sep_HB_pulse(rs):
    lmin, lmax, lmax2   = rs['lmin'], rs['lmax'], rs['lmax2']
    sAC, sDC, sACDC, t  = rs['sAC'], rs['sDC'], rs['sig'], rs['t']
    nlmin, nlmax, pulse = len(lmin), len(lmax), []
    dt = t[1] - t[0]

    for i in range(nlmin-1):
        pulse_peak, pulse_vally, p = [], [], {}
        ii   = i if lmin[i] < lmax[i]  else i+1
        i2   = i if lmin[i] < lmax2[i] else i+1
        sAC0 = sAC[lmin[i]]
        sig  = [ sAC[j] for j in range(lmin[i], lmin[i+1]) ]
        dA1  = [ (sig[j+1]-sig[j])/dt for j in range(len(sig)-1) ]
        aDC  = sDC[lmin[i]:lmin[i+1]].mean()
        for j in range(lmin[i]+2, lmin[i+1]-1):
            if sAC[j+1] > sAC[j] and sAC[j-1] > sAC[j]:
                pulse_vally.append(j)
            if sAC[j+1] < sAC[j] and sAC[j-1] < sAC[j]:
                pulse_peak.append(j)
        p = {
            'idx':    i,
            'sAC':    np.asarray(sig),
            'dA1':    np.asarray(dA1),
            'i0':     lmin[i],
            'iN':     lmin[i+1],
            'iM':     lmax[ii],
            'iM2':    lmax2[i2],
            'Lm':     sACDC[lmin[i]],
            'LM':     sACDC[lmax2[i2]],
            'Ampval': sACDC[lmax2[i2]]-sACDC[lmin[i]],
            'aDC':    aDC,
            'pulse_vally':  pulse_vally.copy(),
            'pulse_peak':   pulse_peak.copy(),
            'npulse_vally': len(pulse_vally),
            'npulse_peak':  len(pulse_peak),
        }
        pulse.append(p)
    return pulse

def get_heart_rate(rs, pulse):
    lmin, lmax, lmax2, t = rs['lmin'], rs['lmax'], rs['lmax2'], rs['t']
    n1, n2, n3, j, j2    = len(lmin), len(lmax), len(lmax2), 0, 0

    while j  < n2 and lmax[j]  <= lmin[0]: j=j+1
    while j2 < n3 and lmax2[j] <= lmin[0]: j2=j2+1
    for i in range(n1-1):
        if i+j < n2 and i+j2 < n3:
            pulse[i]['HBTval'] = t[lmin[i+1]]   - t[lmin[i]]
            pulse[i]['crTa']   = t[lmax[i+j]]   - t[lmin[i]]
            pulse[i]['crTb']   = t[lmax2[i+j2]] - t[lmin[i]]

def get_DNotch_1(t, pulse):
    dt = t[1] - t[0]
    di = int(0.015/dt + 0.5)
    if di <= 0: di=1
    sig, pi0, piM = pulse['sAC'], pulse['i0'], pulse['iM2']
    i0 = piM-pi0+1
    i1 = i0 + int(0.12/dt + 0.5)
    i2 = i0
    iN = int(sig.size*2/3 + 0.5)
    pmin, pmax = sig[0], sig[i0]

    iDPs, vDPs, iDNs, vDNs = [], [], [], []
    while True:
        iDP, vDP, iDN, vDN = 0, pmin, 0, pmax
        for i in range(i1+di, iN-di):
            if sig[i] > sig[i-di] and sig[i] > sig[i+di] and sig[i] > vDP:
                iDP, vDP = i, sig[i]
#        iN2 = np.maximum(i1+di, iN-di) if iDP <= 0 else iDP
        iN2 = iN-di if iDP <= 0 else iDP
        for i in range(i2+di, iN2):
            if sig[i] < sig[i-di] and sig[i] < sig[i+di] and sig[i] < vDN:
                iDN, vDN = i, sig[i]
        iDPs.append(iDP)
        vDPs.append(vDP)
        iDNs.append(iDN)
        vDNs.append(vDN)
        if iDP <= 0: break
        i1 = iDP + int(0.1/dt + 0.5)
        i2 = iDP
    if iDPs[0] < i0:
        if iDNs[0] < i0:
            return False
        else:
            for i in range(iN, sig.size-di):
                if sig[i] > sig[i-di] and sig[i] > sig[i+di]:
                    iDPs[0], vDPs[0] = i, sig[i]
                    break
    ph, k = (pmax-pmin)*0.02, -1
    for i in range(len(iDPs)):
        if vDPs[i]-vDNs[i] >= ph:
            k = i
            break
    if k == -1: return False

    pulse['iDN']    = iDNs[k]+pi0
    pulse['iDP']    = iDPs[k]+pi0
    pulse['DNval']  = sig[iDNs[k]] - pmin
    pulse['DPval']  = sig[iDPs[k]] - pmin
    pulse['tDNval'] = t[iDNs[k]+pi0] - t[pi0]
    pulse['tDPval'] = t[iDPs[k]+pi0] - t[pi0]
    return True

def get_DNotch_2(t, pulse):
    sig, dA1, pi0, piM = pulse['sAC'], pulse['dA1'], pulse['i0'], pulse['iM2']
    dt   = t[1]-t[0]
    pmin = sig[0]
    i0   = piM-pi0+1
    iN   = int(dA1.size * 2/3)

    jNs, vNs = [], []
    while True:
        stat, jN, vN = 0, 0, 0
        for n in range(i0, iN):
            if dt*(n-i0) < 0.075: continue
            dA2 = dA1[n+1] - dA1[n]
            if stat == 0:
                if dA2 < 0: continue
                stat, jN, vN = 1, n, dA2
            else:
                if dA2 < 0 and dt*(n-jN) > 0.05: break
                if dA2 > vN:
                    jN, vN = n, dA2
        jNs.append(jN)
        vNs.append(vN)
        if jN == 0: break
        i0 = jN + int(0.05/dt + 0.5)

    vNmax, k = jNs[0], 0
    for n in range(len(jNs)):
        if vNs[n] > 0.5:
            k = n
            break
        if vNmax < vNs[n]:
            k, vNmax = n, vNs[n]
    i0, stat = jNs[k]+1, 0
    for n in range(i0, sig.size-1):
        dA2 = dA1[n+1] - dA1[n]
        if stat == 0:
            if dA2 > 0: continue
            stat, jP, vP = 1, jNs[k], vNs[k]
        else:
            if dA2 > 0 and dt*(n-jP) > 0.05: break
            if dA2 < vP:
                jP, vP = n, dA2
    i0, vP = jP, sig[jP]
    for n in range(i0, iN):
        if sig[n] > vP:
            jP, vP = n, sig[n]
    if sig[jNs[k]] < sig[jP]:
        for n in range(jNs[k], jP):
            if sig[n] >= vNs[k]: continue
            jNs[k], vNs[k] = n, sig[n]
    iDN = jNs[k] + pi0
    iDP = jP     + pi0
    pulse['iDN']    = iDN
    pulse['iDP']    = iDP
    pulse['DPval']  = sig[iDP-pi0] - pmin
    pulse['DNval']  = sig[iDN-pi0] - pmin
    pulse['tDPval'] = t[iDP] - t[pi0]
    pulse['tDNval'] = t[iDN] - t[pi0]

def get_TailSlope(t, pulse):
    sig, ipeak, slopR = pulse['sAC'], pulse['pulse_peak'], 0.0
    i0   = ipeak[-1] - pulse['i0']
    iN   = sig.size
    di   = int((iN-i0)/6.0 + 0.5)
    if di > 1:
        for j in range(1, 6):
            i  = i0 + di*j
            i1 = i-3 if i-3 > i0 else i0
            i2 = i+3 if i+3 < iN else iN-1
            slopR = slopR + (sig[i2]-sig[i1]) / (t[i2]-t[i1])
    pulse['TailSlope'] = slopR / 5.0

def get_DNotch(rs, pulse):
    t = rs['t']
    for p in pulse:
        if get_DNotch_1(t, p) == False:
            get_DNotch_2(t, p)
        get_TailSlope(t, p)

def get_dA1_peak(para, dA1):
    ndA1  = dA1.size
    pmax  = dA1[0:int(ndA1/4)+1].max()
    imax  = dA1[0:int(ndA1/4)+1].argmax()
    pmin2 = dA1[imax:int(ndA1/3)+1].min()
    imin2 = dA1[imax:int(ndA1/3)+1].argmin() + imax
    v_ave = para['dA1_plat_err'] * pmax

    ip1, ip2, i0arr, iNarr = 0, 0, [], []
    for i in range(imax+1, imin2):
        v1, v2 = dA1[i], dA1[i+1]
        if np.absolute(v2-v1) > v_ave: continue
        for j in range(i+1,imin2+1):
            if j < imin2: v2=dA1[j]
            if np.absolute(v2-v1) > v_ave: break
        i0arr.append(i)
        iNarr.append(j)
    v1 = 0
    for i in range(len(i0arr)):
        v2 = iNarr[i] - i0arr[i]
        if v2 > v1:
            v1, ip1, ip2 = v2, i0arr[i], iNarr[i]
    res = {
        'pmax':  pmax,
        'imax':  imax,
        'imin2': imin2,
        'ip1':   ip1,
        'ip2':   ip2,
    }
    return res

def fit_dA1_peak(para, f, key, t, dA1, res):
    print(f"x0 = {t[res['imax']]:f}",           file=f)
    print(f"s  = {t[res['imax']]/2.0:f}",       file=f)
    print(f"a  = {res['pmax']:f}",              file=f)
    print(f"fit f(x) '-' using 1:2 via a,x0,s", file=f)
    i0 = int(res['imax']/2)
    iN = i0 + res['imax']
    for i in range(i0,iN):
        print(f"{t[i]:f}  {dA1[i]:f}", file=f)
    print(f"e", file=f)
    print(f"print \"{key} \", a, x0, s", file=f);

def analy_dA1(para, rs, pulse):
    fn1 = os.path.join(para['wdir'], "gplot.cmd")
    fn2 = os.path.join(para['wdir'], "fitdA1.log")
    fn3 = "fit.log"
    t   = rs['t']
    if os.path.exists(fn1) == True: os.unlink(fn1)
    if os.path.exists(fn2) == True: os.unlink(fn2)
    if os.path.exists(fn3) == True: os.unlink(fn3)

    try:
        f = open(fn1, "wt")
    except:
        print(f"!!! analy_dA1: cannot output file: {fn1}")
        exit(1)
    print(f"set print '{fn2}'", file=f)
    print(f"f(x) = a * exp(-(x-x0)*(x-x0)/(2*s*s))", file=f)
    for k, p in enumerate(pulse):
        i0, dA1 = p['i0'], p['dA1']
        dA1res  = get_dA1_peak(para, dA1)
        fit_dA1_peak(para, f, f"{k}", t, dA1, dA1res)
        p['pmax']   = dA1res['pmax']
        p['imax']   = dA1res['imax']
        p['imin2']  = dA1res['imin2']
        p['ip1']    = dA1res['ip1']
        p['ip2']    = dA1res['ip2']
        p['ACTa']   = dA1res['pmax'] / p['crTa']
        p['ACTb']   = dA1res['pmax'] / p['crTb']
        p['tip1']   = t[dA1res['ip1']]
        p['tip2']   = t[dA1res['ip2']]
        p['tiplen'] = t[dA1res['ip2']] - t[dA1res['ip1']]
    f.close()
    os.system(f"{para['gplot']} {fn1} > /dev/null 2>&1")

    try:
        f = open(fn2, "rt")
    except:
        print(f"!!! analy_dA1: cannot open file: {fn2}")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.rstrip()
        arr  = line.split()
        k, a, x0 = int(arr[0]), float(arr[1]), float(arr[2])
        s    = np.absolute(float(arr[3]))
        pmax = pulse[k]['pmax']
        tmin = t[pulse[k]['imin2']]
        r    = -1 if a<0 or x0<0 or x0>tmin or a<pmax/2 or a>pmax*2 else 0
        pulse[k]['a']      = a
        pulse[k]['x0']     = x0
        pulse[k]['s']      = s
        pulse[k]['fitres'] = r
    os.unlink(fn1)
    os.unlink(fn2)
    os.unlink(fn3)

def analy_peak_signal(signal, i0, iS, iN, iDN):
    imax, imin, immn = iS-i0, iS-i0, iS-i0
    smax, smin, smmn = signal[iS], signal[iS], signal[iS]
    nsig = signal.size
    if (iN >= nsig): iN=nsig

    for i in range(iS+1, iN):
        if i < iDN:
            if signal[i] > smax:
                imax = i-i0
                smax = signal[i]
            if signal[i] < smin:
                imin = i-i0
                smin = signal[i]
        if signal[i] < smmn:
            immn = i-i0
            smmn = signal[i]
    return imax, imin, immn

def analy_peak_fit(f, signal, t, i0, it, ii, w, key):
    val = signal[i0+ii] if signal[i0+ii] != 0 else 1.0
    i1  = -w if ii >= w else -ii
    print(f"b    = {t[it+ii]:f}", file=f)
    print(f"norm = {val:E}", file=f)
    print(f"fit f(x) '-' using 1:2 via a4,a3,a2,a1", file=f)
    for i in range(i1, w+1):
        print(f"{t[it+ii+i]:f} {signal[i0+ii+i]/val:f}", file=f)
    print(f"e", file=f)
    print(f"print \"{key} \", a4, a3, a2, a1, b, norm", file=f)

def analy_peak(para, rs, pulse):
    fn1  = os.path.join(para['wdir'], "gplot.cmd")
    fn2  = os.path.join(para['wdir'], "fitPeak.log")
    fn3  = "fit.log"
    func = 'f(x) = a4*(x-b)**4 + a3*(x-b)**3 + a2*(x-b)**2 + a1*(x-b) + 1.0'
    if os.path.exists(fn1) == True: os.unlink(fn1)
    if os.path.exists(fn2) == True: os.unlink(fn2)
    if os.path.exists(fn3) == True: os.unlink(fn3)

    try:
        f = open(fn1, "wt")
    except:
        print(f"!!! analy_peak: cannot output file: {fn1}")
        exit(1)
    print(f"set print '{fn2}'", file=f)
    print(f"print '#label   a4   a3   a2   a1   b   norm'", file=f)
    print(f"{func}", file=f)

    t, sig = rs['t'], rs['sAC']
    toff, w0, w1 = para['toff'], para['w0'], para['w1']
    for i, p in enumerate(pulse):
        sdA, i0, iN, iDN = p['dA1'], p['i0'], p['iN'], p['iDN']
        i0_toff = 0 if i0-toff < 0 else i0-toff
        imax0, imin0, immn0 = analy_peak_signal(sig, i0_toff, i0, iN, iDN)
        imax1, imin1, immn1 = analy_peak_signal(sdA, 0, 0, iN-i0, iDN-i0)
        analy_peak_fit(f, sig, t, i0_toff, i0_toff, imax0, w0, f"{i}:a1")
        analy_peak_fit(f, sig, t, i0_toff, i0_toff, imin0, w0, f"{i}:a2")
        analy_peak_fit(f, sdA, t, 0, i0, imax1, w1, f"{i}:a3")
        analy_peak_fit(f, sdA, t, 0, i0, imin1, w1, f"{i}:a4")
        p['imax0'] = imax0
        p['imin0'] = imin0
        p['immn0'] = immn0
        p['imax1'] = imax1
        p['imin1'] = imin1
        p['immn1'] = immn1
    f.close()
    os.system(f"{para['gplot']} {fn1} > /dev/null 2>&1")
    os.unlink(fn3)

    try:
        f = open(fn2, "rt")
    except:
        print(f"!!! analy_peak: cannot open file: {fn2}")
        exit(1)
    lines = f.readlines()
    f.close()
    for line in lines:
        if line[0] == '#': continue
        line = line.rstrip()
        arr  = line.split()
        key, arr = arr[0], np.asarray(arr[1:], dtype=float)
        i, key = key.split(':')
        i = int(i)
        if key == 'a1':
            pulse[i]['a1'] = arr
        elif key == 'a2':
            pulse[i]['a2'] = arr
        elif key == 'a3':
            pulse[i]['a3'] = arr
        else:
            pulse[i]['a4'] = arr
    os.unlink(fn1)
    os.unlink(fn2)

def get_peak_feature(sig, t, i0, it, ii, w0, a):
    res = {
        't':      0,
        'pval':   0,
        'curv':   0,
        'slopeL': 0,
        'slopeR': 0
    }
    nT = len(t)
    if it+ii >= nT: return res

    t0 = t[it+ii]
    t1 = t[it+ii-w0] if it+ii   >= w0 else t[it]
    t2 = t[it+ii+w0] if it+ii+w0 < nT else t[nT-1]
    A  = sig[i0+ii]
    a4, a3, a2, a1 = a[0], a[1], a[2], a[3]

    res['t'], res['pval'] = t0, A
    res['curv'] = 2.0*a2 / np.sqrt(1.0+a1*a1)**3

    slope, dt = 0, (t0-t1)/5.0
    for i in range(5):
        x = t1 + i*dt + 0.5*dt
        slope += (4*a4*(x-t0)**3+3*a3*(x-t0)**2+2*a2*(x-t0)+a1)
    res['slopeL'] = slope/5.0

    slope, dt = 0, (t2-t0)/5.0
    for i in range(5):
        x = t0 + i*dt + 0.5*dt
        slope += (4*a4*(x-t0)**3+3*a3*(x-t0)**2+2*a2*(x-t0)+a1)
    res['slopeR'] = slope/5.0
    return res

def peak_feature(para, rs, pulse):
    t, sig = rs['t'], rs['sAC'],
    toff, w0, w1 = para['toff'], para['w0'], para['w1']
    for p in pulse:
        sdA,  i0                = p['dA1'],  p['i0']
        imax0,imin0,imax1,imin1 = p['imax0'],p['imin0'],p['imax1'],p['imin1']
        a1,   a2,   a3,   a4    = p['a1'],   p['a2'],   p['a3'],   p['a4']
        ishft = 0 if i0-toff < 0 else i0-toff

        p['res1'] = get_peak_feature(sig, t, ishft, ishft, imax0, w0, a1)
        p['res2'] = get_peak_feature(sig, t, ishft, ishft, imin0, w0, a2)
        p['res3'] = get_peak_feature(sdA, t, 0, i0, imax1, w1, a3)
        p['res4'] = get_peak_feature(sdA, t, 0, i0, imin1, w1, a4)
        if p['res1']['t'] > 0:
            p['res1']['t']    -= t[i0]
            p['res1']['pval'] -= sig[i0]
        if p['res2']['t'] > 0:
            p['res2']['t']    -= t[i0]
            p['res2']['pval'] -= sig[i0]
        if p['res3']['t'] > 0:
            p['res3']['t']    -= t[i0]
            p['res3']['pval'] -= sdA[0]
        if p['res4']['t'] > 0:
            p['res4']['t']    -= t[i0]
            p['res4']['pval'] -= sdA[0]

def PPG_aveerr(n, pnum, label, pulse):
    pdata = []
    if pnum == 'NULL':
        for i in range(n):
            pdata.append(pulse[i][label])
    else:
        for i in range(n):
            pdata.append(pulse[i][pnum][label])
    pdata = np.asarray(pdata)
    ave   = np.mean(pdata)
    err   = np.std(pdata)/np.sqrt(n-1) if n > 1 else 0.0
    return ave, err

def PPG_feature_ave(para, pulse):
    n, pnum, ares = len(pulse), 'NULL', {}
    ares['aDP'],     ares['dDP']      = PPG_aveerr(n, pnum, 'DPval',  pulse)
    ares['aDN'],     ares['dDN']      = PPG_aveerr(n, pnum, 'DNval',  pulse)
    ares['aLm'],     ares['dLm']      = PPG_aveerr(n, pnum, 'Lm',     pulse)
    ares['aLM'],     ares['dLM']      = PPG_aveerr(n, pnum, 'LM',     pulse)
    ares['aveAmp'],  ares['errAmp']   = PPG_aveerr(n, pnum, 'Ampval', pulse)
    ares['aveDC'],   ares['errDC']    = PPG_aveerr(n, pnum, 'aDC',    pulse)
    ares['tDP'],     ares['dtDP']     = PPG_aveerr(n, pnum, 'tDPval', pulse)
    ares['tDN'],     ares['dtDN']     = PPG_aveerr(n, pnum, 'tDNval', pulse)
    ares['CresTa'],  ares['dCresTa']  = PPG_aveerr(n, pnum, 'crTa',   pulse)
    ares['CresTb'],  ares['dCresTb']  = PPG_aveerr(n, pnum, 'crTb',   pulse)
    ares['HBT'],     ares['dHBT']     = PPG_aveerr(n, pnum, 'HBTval', pulse)
    ares['dA1_a'],   ares['dA1_da']   = PPG_aveerr(n, pnum, 'a',      pulse)
    ares['dA1_x0'],  ares['dA1_dx0']  = PPG_aveerr(n, pnum, 'x0',     pulse)
    ares['dA1_s'],   ares['dA1_ds']   = PPG_aveerr(n, pnum, 's',      pulse)
    ares['dA1_ACTa'],ares['dA1_dACTa']= PPG_aveerr(n, pnum, 'ACTa',   pulse)
    ares['dA1_ACTb'],ares['dA1_dACTb']= PPG_aveerr(n, pnum, 'ACTb',   pulse)
    ares['dA1_pt1'], ares['dA1_dpt1'] = PPG_aveerr(n, pnum, 'tip1',   pulse)
    ares['dA1_pt2'], ares['dA1_dpt2'] = PPG_aveerr(n, pnum, 'tip2',   pulse)
    ares['dA1_plen'],ares['dA1_dplen']= PPG_aveerr(n, pnum, 'tiplen', pulse)
    ares['aPvally'], ares['dPvally']  = PPG_aveerr(n, pnum,'npulse_vally',pulse)
    ares['aPpeak'],  ares['dPpeak']   = PPG_aveerr(n, pnum,'npulse_peak', pulse)
    ares['aTailSlope'],ares['dTailSlope'] = PPG_aveerr(n,pnum,'TailSlope',pulse)
    for pnum in ['res1', 'res2', 'res3', 'res4']:
        pres = {}
        pres['t'],      pres['dt']      = PPG_aveerr(n, pnum, 't',      pulse)
        pres['Amp'],    pres['dAmp']    = PPG_aveerr(n, pnum, 'pval',   pulse)
        pres['curv'],   pres['dcurv']   = PPG_aveerr(n, pnum, 'curv',   pulse)
        pres['slopeL'], pres['dslopeL'] = PPG_aveerr(n, pnum, 'slopeL', pulse)
        pres['slopeR'], pres['dslopeR'] = PPG_aveerr(n, pnum, 'slopeR', pulse)
        ares[f"p{pnum}"] = pres.copy()
    return ares

#--------------------------------------------------------------------------
#	PPG signal analysis main subroutine.
#
def PPG_sig_analy(para, data):
    ok  = False
    res = []
    print("* Analyze PPG signals ....")
    for signame in ['IR', 'RD']:
        rs, ok = PPG_preproc(para, data, signame)
        if ok != True: break
        res.append(rs)
    if ok == True:
        for i in range(len(res)):
            rs = res[i]
            print(f"    {data['label']}: {data['name']}, {rs['signame']} ....")
            pulse = sep_HB_pulse(rs)
            get_heart_rate(rs, pulse)
            get_DNotch(rs, pulse)
            analy_dA1(para, rs, pulse)
            analy_peak(para, rs, pulse)
            peak_feature(para, rs, pulse)
            ares = PPG_feature_ave(para, pulse)
            rs['pulse'] = pulse
            rs['ave']   = ares
        data['res'] = res
    else:
        data['res'] = None
    data['sigok'] = ok


#--------------------------------------------------------------------------
#	Read input parameters.
#
def get_gnuplot():
    fn = '/usr/bin/gnuplot'
    if os.access(fn, os.X_OK) == True: return fn
    fn = '/usr/local/bin/gnuplot'
    if os.access(fn, os.X_OK) == True: return fn
    fn = '/sw/bin/gnuplot'
    if os.access(fn, os.X_OK) == True: return fn
    fn = '/opt/local/bin/gnuplot'
    if os.access(fn, os.X_OK) == True: return fn
    print("!!! Gnuplot is not available in this system")
    exit(1)
   
def inputs():
    inpf = 'ppg_input.txt'

    para = {
        'outdir':       'out',
        'ttot':         60,
        'interp_algo':  2,
        'dsample':      250,
        'runPPGL':      'yes',
        'runPPGR':      'yes',
        'OCR1':         0,
        'OCR2':         0,
        'T_step':       60,
        'T_init_skip':  3,
        'T_dt_stepb':   0.01,
        'AC_Hz':        0.5,
        'NZ_Hz':        10,
        'TH_Ratio':     0.25,
        'dA1_plat_ave': 0.1,
        'dA1_plat_err': 0.1,
        'PPG_Norm':     1,
        'wdir':         'PPG_WORK',
    }
    try:
        f = open(inpf, 'r')
    except:
        print(f"!!! Cannot open file: {inpf}")
        exit(1)
    lines = f.readlines()
    f.close()

    for line in lines:
        line = line.rstrip()
        if line.find('#') >= 0:
            line = line[:line.find('#')]
        if len(line) == 0: continue

        arr = line.split()
        if arr[0] == 'OUT_DIR:':
            para['outdir'] = arr[1]
        elif arr[0] == 'DATA_T_TOT:':
            para['ttot'] = float(arr[1])
        elif arr[0] == 'DATA_INTERP_ALGO:':
            para['interp_algo'] = int(arr[1])
        elif arr[0] == 'DATA_SAMPLE_R:':
            para['dsample'] = int(arr[1])
        elif arr[0] == 'run_PPG_L:':
            para['runPPGL'] = arr[1]
        elif arr[0] == 'run_PPG_R:':
            para['runPPGR'] = arr[1]
        elif arr[0] == 'Outlier_Cut_Ratio_1:':
            para['OCR1'] = float(arr[1])
        elif arr[0] == 'Outlier_Cut_Ratio_2:':
            para['OCR2'] = float(arr[1])

        elif arr[0] == 'T_STEP:':
            para['T_step'] = float(arr[1])
        elif arr[0] == 'T_INIT_SKIP:':
            para['T_init_skip'] = float(arr[1])
        elif arr[0] == 'T_DT_STEPB:':
            para['T_dt_stepb'] = float(arr[1])
        elif arr[0] == 'AC_HZ:':
            para['AC_Hz'] = float(arr[1])
        elif arr[0] == 'NZ_HZ:':
            para['NZ_Hz'] = float(arr[1])
        elif arr[0] == 'TH_Ratio:':
            para['TH_Ratio'] = float(arr[1])
        elif arr[0] == 'DA1_PLAT_AVE:':
            para['dA1_plat_ave'] = float(arr[1])
        elif arr[0] == 'DA1_PLAT_ERR:':
            para['dA1_plat_err'] = float(arr[1])
        elif arr[0] == 'PPG_Norm:':
            para['PPG_Norm'] = int(arr[1])
    para['ttot'] -= para['T_init_skip']
    para['gplot'] = get_gnuplot()
    para['toff']  = 30
    para['w0']    = 15
    para['w1']    = 10

    return para

#--------------------------------------------------------------------------
#	Main program
#
if __name__ == "__main__":
    para  = inputs()
    infos = read_subinfo(para)

    for info in infos:
        dsets = check_subinfo(para, info)
        if len(dsets) == 0: continue

        for dset in dsets:
            if read_src(para, dset) != True: continue
            PPG_sig_analy(para, dset)
            out_AC_DC_data(para, dset)
            if dset['sigok'] == True:
                out_min_max(para, dset)
                out_dA1_res(para, dset)
                out_peak_feature(para, dset)
                out_PPG_sigFT(para, dset)
                out_ave(para, dset)
        for dset in dsets:
            move_outfiles(dset)
    try:
        os.rmdir(para['wdir'])
    except:
        print(f"Data files still remain in {para['wdir']}")
