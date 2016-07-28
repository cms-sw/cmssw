from __future__ import print_function
import os, sys, math
import numpy as np

import ROOT
ROOT.TH1.AddDirectory(False)

#Try loading rootpy (installed via anaconda)
try:
    import rootpy
    import rootpy.io
    ROOTPY = True
except Exception as e:
    ROOTPY = False
from TTH.MEAnalysis.samples_base import getSitePrefix

trigbits = []
sourceobjects = "selLeptons"

class Lepton:
    def __init__(self, **kwargs):
        self.pt = kwargs.get("pt")
        self.eta = kwargs.get("eta")
        self.phi = kwargs.get("phi")
        self.mass = kwargs.get("mass")
        self.pdgId = kwargs.get("pdgId")

def lepton_selection(lepton):
    return True

triggers_SL_e = [
    "HLT_Ele27_eta2p1_WPTight_Gsf_v"
]

# Nota: ttH triggers are Mu22
# but not available, so use Mu27
# https://github.com/jpata/tthbb-sync/blob/master/Sync16.md#trigger-selection
triggers_SL_m = [
    "HLT_IsoMu27_v",
    "HLT_IsoTkMu27_v",
]

triggers_DL_mumu = [
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v"
]

triggers_DL_elmu = [
    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v",
    "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v",
]

triggers_DL_elel = [
    "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v",
]

def check_triggerbit(row, name):
    prefs = ["HLT_BIT_", "HLT2_BIT_"]
    for pref in prefs:
        bit = getattr(row, pref+name, -1)
        if bit != -1:
            return bit

def check_triggers_OR(row, triggers):
    vals = np.array([check_triggerbit(row, t) for t in triggers])
    return np.any(vals==1)

class Event:
    def __init__(self, event):
        self.leptons = []
        for nlep in range(event.nselLeptons):
            lep = Lepton(
                pt=event.selLeptons_pt[nlep],
                eta=event.selLeptons_eta[nlep],
                phi=event.selLeptons_phi[nlep],
                mass=event.selLeptons_mass[nlep],
                pdgId=event.selLeptons_pdgId[nlep],
            )
            self.leptons += [lep]
        
        self.leptons = filter(lepton_selection, self.leptons)
        self.is_sl = len(self.leptons) == 1
        self.is_dl = len(self.leptons) == 2
        self.pass_trig_SL_mu = check_triggers_OR(event, triggers_SL_m)
        self.pass_trig_SL_el = check_triggers_OR(event, triggers_SL_e)
        
        self.pass_trig_DL_mumu = check_triggers_OR(event, triggers_DL_mumu)
        self.pass_trig_DL_elmu = check_triggers_OR(event, triggers_DL_elmu)
        self.pass_trig_DL_elel = check_triggers_OR(event, triggers_DL_elel)

class Fillable(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.selection = kwargs.get("selection")
        self.coords = kwargs.get("coords")
        self.outdir = kwargs.get("outdir", ROOT.gROOT)

def make_binarray(bins):
    vec = getattr(ROOT, "std::vector<double>")()
    for b in bins:
        vec.push_back(b)
    return vec

class Fillable1(Fillable):
    def __init__(self, **kwargs):
        super(Fillable1, self).__init__(**kwargs)
        print("creating {0}".format(self.name))

        binsx = make_binarray(kwargs.get("binsx"))
        binsy = make_binarray(kwargs.get("binsy"))
        self.hist = ROOT.TH2D(
            self.name, self.name,
            binsx.size()-1, binsx.data(),
            binsy.size()-1, binsy.data(),
        )
        self.objs = [self.hist]
        self.outdir.Add(self.hist)

    def fill(self, event):
        if self.selection(event):
            coord1, coord2 = self.coords(event)
            self.hist.Fill(coord1, coord2)

class Fillable2(Fillable):
    def __init__(self, **kwargs):
        super(Fillable2, self).__init__(**kwargs)
        print("creating {0}".format(self.name))
        
        binsx1 = make_binarray(kwargs.get("binsx"))
        binsy1 = make_binarray(kwargs.get("binsy"))
        binsx2 = make_binarray(kwargs.get("binsx"))
        binsy2 = make_binarray(kwargs.get("binsy"))
        
        self.hist1 = ROOT.TH2D(
            self.name+"1", self.name+"1",
            binsx1.size()-1, binsx1.data(),
            binsy1.size()-1, binsy1.data(),
        )
        self.hist2 = ROOT.TH2D(
            self.name+"2", self.name+"2",
            binsx2.size()-1, binsx2.data(),
            binsy2.size()-1, binsy2.data(),
        )
        self.objs = [self.hist1, self.hist2]
        self.outdir.Add(self.hist1)
        self.outdir.Add(self.hist2)

    def fill(self, event):
        if self.selection(event):
            c1, c2 = self.coords(event)
            self.hist1.Fill(*c1)
            self.hist2.Fill(*c2)

class FillPair(object):

    def __init__(self, class_fillable, kwargs1, kwargs2, kwargs, outdir):
        self.kwargs1 = kwargs1
        self.kwargs2 = kwargs2

        self.kwargs1.update(kwargs)
        self.kwargs2.update(kwargs)
        self.kwargs2["selection"] = lambda ev, sel1=self.kwargs1["selection"], sel2=self.kwargs2["selection"]: sel1(ev) and sel2(ev)
        self.kwargs1["outdir"] = outdir
        self.kwargs2["outdir"] = outdir

        self.h1 = class_fillable(**kwargs1)
        self.h2 = class_fillable(**kwargs2)

    def fill(self, event):
        self.h1.fill(event)
        self.h2.fill(event)

nbins_pt = 100
nbins_eta = 100

if __name__ == "__main__":
    if os.environ.has_key("FILE_NAMES"):
        file_names = map(getSitePrefix, os.environ["FILE_NAMES"].split())
    else:
        file_names = ["root://storage01.lcg.cscs.ch//pnfs/lcg.cscs.ch/cms/trivcat/store"+
            "/user/jpata/tth/Jul15_leptonic_v1/ttHTobb_M125_13TeV_powheg_pythia8" +
            "/Jul15_leptonic_v1/160715_182411/0000/tree_{0}.root".format(i)
            for i in range(1, 51)
        ]
    for fn in file_names:
        print(fn)

    outfile = ROOT.TFile("out.root", "RECREATE")

    histos = {}
    histos["IsoMu22_OR_IsoTkMu22"] = FillPair(
        Fillable1,
        {"name": "mu_all", "selection": lambda ev: ev.is_sl and abs(ev.leptons[0].pdgId)==13},
        {"name": "mu_IsoMu27_OR_IsoTkMu27", "selection": lambda ev: ev.pass_trig_SL_mu},
        {
            "coords": lambda ev: (ev.leptons[0].pt, ev.leptons[0].eta),
            "binsx": [0, 10, 15, 18, 22, 24, 26, 30, 40, 50, 60, 80, 120, 200, 500],
            "binsy": [-2.4, -2.1, -1.6, -1.2, -0.9, -0.3, -0.2, 0.2, 0.3, 0.9, 1.2, 1.6, 2.1, 2.4]
        },
        outfile
    )

    for file_name in file_names:
        tf = ROOT.TFile.Open(file_name)
        branches = ["selLeptons", "nselLeptons", "HLT", "HLT2", "Vtype"]
        if ROOTPY:
            events = rootpy.asrootpy(tf.Get("vhbb/tree"))
            events.deactivate("*")
            for br in branches:
                events.activate(br+"*")
        else:
            events = tf.Get("vhbb/tree")
            events.SetBranchStatus("*", False)
            for br in branches:
                events.SetBranchStatus(br+"*", True)

        for row in events:
            if row.nselLeptons == 0:
                continue

            event = Event(row)

            for (k, v) in histos.items():
                v.fill(event)
        
        tf.Close()
    print("writing output")
    outfile.Write() 
    outfile.Close()
