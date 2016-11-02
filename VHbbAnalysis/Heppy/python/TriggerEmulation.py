import ROOT
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer

class Calibration1:
    def __init__(self, tfile, h1, h2):
        self.tfile = tfile
        self.h_all = self.tfile.Get(h1)
        self.h_trg = self.tfile.Get(h2)
        if not self.h_all or not self.h_trg:
            raise Exception("Didn't find input histograms {0} and {1} in the file".format(h1, h2))

    def getValue(self, lepton):
        pt, eta = lepton.pt(), lepton.eta()

        ibin = self.h_all.FindBin(pt, eta)
        n1 = self.h_all.GetBinContent(ibin)
        n2 = self.h_trg.GetBinContent(ibin)
        return float(n2)/float(n1) if n1>0 else -1.0

class Calibration2:
    def __init__(self, tfile, h1, h2):
        self.c1 = Calibration1(tfile, h1+"1", h2+"1")
        self.c2 = Calibration1(tfile, h1+"2", h2+"2")

    def getValue(self, leptons):
        v1 = self.c1.getValue(leptons[0])
        v2 = self.c2.getValue(leptons[1])
        return v1*v2

class TriggerEmulationAnalyzer(Analyzer):
     
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(TriggerEmulationAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        
        self.calib_file = ROOT.TFile(self.cfg_ana.calibrationFile)
        if not self.calib_file:
            raise Exception("Could not open input file: {0}".format(self.cfg_ana.calibrationFile))

        self.calibrations = {
            "mu": Calibration1(self.calib_file, "mu_fine_all", "mu_fine_hlt"),
            "el": Calibration1(self.calib_file, "el_fine_all", "el_fine_hlt"),
            "elel": Calibration2(self.calib_file, "elel_fine_all", "elel_fine_hlt"),
            "mumu": Calibration2(self.calib_file, "mumu_fine_all", "mumu_fine_hlt"),
            "elmu": Calibration2(self.calib_file, "elmu_fine_all", "elmu_fine_hlt"),
        }

    def process(self, event):
        
        slElectrons = [x for x in event.selectedElectrons if self.cfg_ana.slEleSelection(x) ]
        slMuons = [x for x in event.selectedMuons if self.cfg_ana.slMuSelection(x) ]
        dlElectrons = [x for x in event.selectedElectrons if self.cfg_ana.dlEleSelection(x) ]
        dlMuons = [x for x in event.selectedMuons if self.cfg_ana.dlMuSelection(x) ]
        dlLeptons = sorted(dlElectrons + dlMuons, key=lambda x: x.pt(), reverse=True)

        is_sl_el = len(slElectrons) == 1 and len(dlLeptons)==1
        is_sl_mu = len(slMuons) == 1 and len(dlLeptons)==1
        is_dl_mumu = len(dlMuons)==2 and len(dlElectrons)==0
        is_dl_elel = len(dlMuons)==0 and len(dlElectrons)==2
        is_dl_elmu = len(dlMuons)==1 and len(dlElectrons)==1
       
        event.triggerEmulationWeight = 1.0
        kind = ""
        if is_sl_mu:
            lep = slMuons[0]
            event.triggerEmulationWeight = self.calibrations["mu"].getValue(lep)
            kind = "sl_mu"
        elif is_sl_el:
            lep = slElectrons[0]
            event.triggerEmulationWeight = self.calibrations["el"].getValue(lep)
            kind = "sl_el"
        elif is_dl_elel:
            event.triggerEmulationWeight = self.calibrations["elel"].getValue(dlLeptons)
            kind = "dl_elel"
        elif is_dl_elmu:
            event.triggerEmulationWeight = self.calibrations["elmu"].getValue(dlLeptons)
            kind = "dl_elmu"
        elif is_dl_mumu:
            event.triggerEmulationWeight = self.calibrations["mumu"].getValue(dlLeptons)
            kind = "dl_mumu"
        
        return True
