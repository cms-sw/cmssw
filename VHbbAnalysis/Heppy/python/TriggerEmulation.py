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
        return float(n2)/float(n1)

class TriggerEmulationAnalyzer(Analyzer):
     
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(TriggerEmulationAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        
        self.calib_file = ROOT.TFile(self.cfg_ana.calibrationFile)
        if not self.calib_file:
            raise Exception("Could not open input file: {0}".format(self.cfg_ana.calibrationFile))

        self.calibrations = {
            "mu": Calibration1(self.calib_file, "mu_fine_all", "mu_fine_hlt")
        }

    def process(self, event):
        
        slElectrons = [x for x in event.selectedElectrons if self.cfg_ana.slEleSelection(x) ]
        slMuons = [x for x in event.selectedMuons if self.cfg_ana.slMuSelection(x) ]
        dlElectrons = [x for x in event.selectedElectrons if self.cfg_ana.dlEleSelection(x) ]
        dlMuons = [x for x in event.selectedMuons if self.cfg_ana.dlMuSelection(x) ]
        dlLeptons = dlElectrons + dlMuons

        is_sl_el = len(slElectrons) == 1 and len(dlLeptons)==1
        is_sl_mu = len(slMuons) == 1 and len(dlLeptons)==1
       
        event.triggerEmulationWeight = 1.0
        if is_sl_mu:
            lep = slMuons[0]
            event.triggerEmulationWeight = self.calibrations["mu"].getValue(lep)
        return True
