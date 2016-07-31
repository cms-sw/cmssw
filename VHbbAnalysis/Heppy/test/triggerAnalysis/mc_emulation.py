from __future__ import print_function
import os, sys, math
import numpy as np

import ROOT
ROOT.TH1.AddDirectory(False)
ROOT.TH1.SetDefaultSumw2(True)

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
    branches = [
        "pt", "eta", "phi", "mass",
        "pdgId",
        "tightId", "looseIdPOG", "pfRelIso04",
        "etaSc", "eleSieie", "eleHoE", "eleEcalClusterIso", "eleHcalClusterIso", "dr03TkSumPt", "eleDEta",
        "eleMVAIdSpring15Trig"
    ]
    def __init__(self, row, prefix, idx):
        for br in self.branches:
            setattr(self, br, getattr(row, prefix+br)[idx])

def lepton_selection(lepton):
    return True

triggers_SL_e = [
    "HLT_Ele27_eta2p1_WPTight_Gsf_v"
]

# Nota: ttH triggers are Mu22
# but not available, so use Mu27
# https://github.com/jpata/tthbb-sync/blob/master/Sync16.md#trigger-selection
triggers_SL_m = [
    "HLT_IsoMu22_v",
    "HLT_IsoTkMu22_v",
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
            
bins_pt_coarse = [0, 10, 15, 18, 22, 24, 26, 30, 40, 50, 60, 80, 120, 200, 500]
bins_pt_fine = [0,10] + [k for k in np.arange(15, 30, 0.1)] + [30, 40, 50, 60, 80, 120, 200, 500]
bins_eta = [-2.4, -2.1, -1.6, -1.2, -0.9, -0.3, -0.2, 0.2, 0.3, 0.9, 1.2, 1.6, 2.1, 2.4]
def check_triggerbit(row, name):
    prefs = ["HLT_BIT_", "HLT2_BIT_"]
    for pref in prefs:
        bit = getattr(row, pref+name, -1)
        if bit != -1:
            return bit

def check_triggers_OR(row, triggers):
    vals = np.array([check_triggerbit(row, t) for t in triggers])
    return np.any(vals==1)

# unfortunately we have to copy this function manually
# https://github.com/vhbb/cmssw/blob/vhbbHeppy80X/VHbbAnalysis/Heppy/test/vhbb.py#L305
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/MultivariateElectronIdentificationRun2#Recommended_MVA_recipes_for_2015
def ele_mvaEleID_Trig_preselection(ele) : 
    return (ele.pt>15 and 
        ( ( abs(ele.etaSc) < 1.4442 and ele.eleSieie < 0.012 and ele.eleHoE < 0.09 and (ele.eleEcalClusterIso / ele.pt) < 0.37 and (ele.eleHcalClusterIso / ele.pt) < 0.25 and (ele.dr03TkSumPt / ele.pt) < 0.18 and abs(ele.eleDEta) < 0.0095 and abs(ele.eleDEta) < 0.065 ) or 
          ( abs(ele.etaSc) > 1.5660 and ele.eleSieie < 0.033 and ele.eleHoE <0.09 and (ele.eleEcalClusterIso / ele.pt) < 0.45 and (ele.eleHcalClusterIso / ele.pt) < 0.28 and (ele.dr03TkSumPt / ele.pt) < 0.18 ) ) )


class Event:
    def __init__(self, event):
        self.leptons = []
        for nlep in range(event.nselLeptons):
            lep = Lepton(
                event,
                "selLeptons_",
                nlep
            )
            self.leptons += [lep]
        
        self.leptons = filter(lepton_selection, self.leptons)
        mu = filter(lambda x: abs(x.pdgId) == 13, self.leptons)
        el = filter(lambda x: abs(x.pdgId) == 11, self.leptons)

        #lepton defs from https://github.com/vhbb/cmssw/blob/vhbbHeppy80X/VHbbAnalysis/Heppy/test/vhbb.py#L316
        self.mu_tight = filter(lambda x: x.tightId and x.pfRelIso04 < 0.15, mu)
        self.mu_loose = filter(lambda x: x.looseIdPOG and x.pfRelIso04 < 0.25, mu)
        self.el_tight = filter(
            lambda x: x.eleMVAIdSpring15Trig==2 and ele_mvaEleID_Trig_preselection(x),
            el 
        )
        self.el_loose = filter(
            lambda x: x.eleMVAIdSpring15Trig>=1 and ele_mvaEleID_Trig_preselection(x),
            el 
        )
        self.leptons_loose = sorted(self.mu_loose + self.el_loose, key=lambda x: x.pt, reverse=True)
        self.leptons_tight = sorted(self.mu_tight + self.el_tight, key=lambda x: x.pt, reverse=True)
        
        self.is_sl = len(self.leptons_tight) == 1 and len(self.leptons_loose) == 1
        self.is_dl = not self.is_sl and len(self.leptons_loose) == 2

        self.pass_trig_SL_mu = check_triggers_OR(event, triggers_SL_m)
        self.pass_trig_SL_el = check_triggers_OR(event, triggers_SL_e)
        
        self.pass_trig_DL_mumu = check_triggers_OR(event, triggers_DL_mumu)
        self.pass_trig_DL_elmu = check_triggers_OR(event, triggers_DL_elmu)
        self.pass_trig_DL_elel = check_triggers_OR(event, triggers_DL_elel)
        
        self.triggerEmulationWeight = getattr(event, "triggerEmulationWeight", -1.0)

class Fillable(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.selection = kwargs.get("selection")
        self.weight = kwargs.get("weight", lambda ev: 1.0)
        self.coords = kwargs.get("coords")
        self.outdir = kwargs.get("outdir", ROOT.gROOT)

def make_binarray(bins):
    vec = getattr(ROOT, "std::vector<double>")()
    for b in list(bins):
        vec.push_back(float(b))
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
            w = self.weight(event)
            self.hist.Fill(coord1, coord2, w)

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
            w = self.weight(event)
            c1, c2 = self.coords(event)
            self.hist1.Fill(c1[0], c1[1], w)
            self.hist2.Fill(c2[0], c2[1], w)

class FillPair(object):

    def __init__(self, class_fillable, kwargs1, kwargs2, kwargs3, kwargs, outdir):
        self.kwargs1 = kwargs1
        self.kwargs2 = kwargs2
        self.kwargs3 = kwargs3

        self.kwargs1.update(kwargs)
        self.kwargs2.update(kwargs)
        self.kwargs3.update(kwargs)
        self.kwargs1["outdir"] = outdir
        self.kwargs2["outdir"] = outdir
        self.kwargs3["outdir"] = outdir

        self.h1 = class_fillable(**kwargs1)
        self.h2 = class_fillable(**kwargs2)
        self.h3 = class_fillable(**kwargs3)

    def fill(self, event):
        self.h1.fill(event)
        self.h2.fill(event)
        self.h3.fill(event)

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
    histos["mu_fine"] = FillPair(
        Fillable1,
        {"name": "mu_fine_all", "selection": lambda ev: ev.is_sl and len(ev.mu_tight)==1},
        {"name": "mu_fine_hlt", "selection": lambda ev: ev.is_sl and len(ev.mu_tight)==1 and ev.pass_trig_SL_mu},
        {"name": "mu_fine_emu", "selection": lambda ev: ev.is_sl and len(ev.mu_tight)==1, "weight": lambda ev: ev.triggerEmulationWeight},
        {
            "coords": lambda ev: (ev.mu_tight[0].pt, ev.mu_tight[0].eta),
            "binsx": bins_pt_fine,
            "binsy": bins_eta,
        },
        outfile
    )
    
    histos["el_fine"] = FillPair(
        Fillable1,
        {"name": "el_fine_all", "selection": lambda ev: ev.is_sl and len(ev.el_tight)==1},
        {"name": "el_fine_hlt", "selection": lambda ev: ev.is_sl and len(ev.el_tight)==1 and ev.pass_trig_SL_el},
        {"name": "el_fine_emu", "selection": lambda ev: ev.is_sl and len(ev.el_tight)==1, "weight": lambda ev: ev.triggerEmulationWeight},
        {
            "coords": lambda ev: (ev.el_tight[0].pt, ev.el_tight[0].eta),
            "binsx": bins_pt_fine,
            "binsy": bins_eta, 
        },
        outfile
    )
    
    histos["mumu_fine"] = FillPair(
        Fillable2,
        {"name": "mumu_fine_all", "selection": lambda ev: ev.is_dl and len(ev.mu_loose)==2},
        {"name": "mumu_fine_hlt", "selection": lambda ev: ev.is_dl and len(ev.mu_loose)==2 and ev.pass_trig_DL_mumu},
        {"name": "mumu_fine_emu", "selection": lambda ev: ev.is_dl and len(ev.mu_loose)==2, "weight":  lambda ev: ev.triggerEmulationWeight},
        {
            "coords": lambda ev: ((ev.leptons_loose[0].pt, ev.leptons_loose[0].eta), (ev.leptons_loose[1].pt, ev.leptons_loose[1].eta)),
            "binsx": bins_pt_fine,
            "binsy": bins_eta, 
        },
        outfile
    )

    histos["elel_fine"] = FillPair(
        Fillable2,
        {"name": "elel_fine_all", "selection": lambda ev: ev.is_dl and len(ev.el_loose)==2},
        {"name": "elel_fine_hlt", "selection": lambda ev: ev.is_dl and len(ev.el_loose)==2 and ev.pass_trig_DL_elel},
        {"name": "elel_fine_emu", "selection": lambda ev: ev.is_dl and len(ev.el_loose)==2, "weight":  lambda ev: ev.triggerEmulationWeight},
        {
            "coords": lambda ev: ((ev.leptons_loose[0].pt, ev.leptons_loose[0].eta), (ev.leptons_loose[1].pt, ev.leptons_loose[1].eta)),
            "binsx": bins_pt_fine,
            "binsy": bins_eta, 
        },
        outfile
    )
    
    histos["elmu_fine"] = FillPair(
        Fillable2,
        {"name": "elmu_fine_all", "selection": lambda ev: ev.is_dl and len(ev.el_loose)==1 and len(ev.mu_loose)==1},
        {"name": "elmu_fine_hlt", "selection": lambda ev: ev.is_dl and len(ev.el_loose)==1 and len(ev.mu_loose)==1 and ev.pass_trig_DL_elmu},
        {"name": "elel_fine_emu", "selection": lambda ev: ev.is_dl and len(ev.el_loose)==2, "weight":  lambda ev: ev.triggerEmulationWeight},
        {
            "coords": lambda ev: ((ev.leptons_loose[0].pt, ev.leptons_loose[0].eta), (ev.leptons_loose[1].pt, ev.leptons_loose[1].eta)),
            "binsx": bins_pt_fine,
            "binsy": bins_eta, 
        },
        outfile
    )

    for file_name in file_names:
        tf = ROOT.TFile.Open(file_name)
        branches = ["selLeptons", "nselLeptons", "HLT", "HLT2", "Vtype", "triggerEmulationWeight"]
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
