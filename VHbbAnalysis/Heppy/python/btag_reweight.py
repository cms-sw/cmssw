# Script to test the btag reweighting

# Define a dummy class of a jet object
class Jet :
    def __init__(self, pt, eta, fl, csv, csvname) :
        self.pt = pt
        self.eta = eta
        self.hadronFlavour = fl
        setattr(self, csvname, csv)

from PhysicsTools.Heppy.physicsutils.BTagWeightCalculator import BTagWeightCalculator
import os

#Set up offline b-weight calculation
csvpath = os.environ['CMSSW_BASE']+"/src/VHbbAnalysis/Heppy/data/csv"
bweightcalc = BTagWeightCalculator(
    csvpath + "/csv_rwt_fit_hf_76x_2016_02_08.root",
    csvpath + "/csv_rwt_fit_lf_76x_2016_02_08.root"
)
bweightcalc.btag = "pfCombinedInclusiveSecondaryVertexV2BJetTags"


# EXAMPLE (1): per-jet nominal weight
print "Example (1): per-jet nominal weight"
jet = Jet(50., 1.2, 5, 0.89, bweightcalc.btag)
jet_weight_nominal = bweightcalc.calcJetWeight(
    jet, kind="final", systematic="nominal",
    )
print "\tnominal jet weight:", jet_weight_nominal

# EXAMPLE (2): per-jet systematic up/down weight
print "Example (2): per-jet systematic up/down weight"
for syst in ["JES", "LF", "HF", "LFStats1", "LFStats2", "HFStats1", "HFStats2", "cErr1", "cErr2"]:
    for sdir in ["Up", "Down"]:
        jet_weight_shift = bweightcalc.calcJetWeight(
            jet, kind="final", systematic=syst+sdir
            )
        print "\tsystematic:", syst+sdir, ":", jet_weight_shift

# EXAMPLE (3): the nominal event weight 
print "Example (3): the nominal event weight"
jet1 = Jet(50., -1.2, 5, 0.99, bweightcalc.btag)
jet2 = Jet(30., 1.8, 4, 0.2, bweightcalc.btag)
jet3 = Jet(100., 2.2, 0, 0.1, bweightcalc.btag)
jet4 = Jet(20., 0.5,-5, 0.6, bweightcalc.btag)
jets = [jet1,jet2,jet3,jet4]
event_weight_nominal = bweightcalc.calcEventWeight(
    jets, kind="final", systematic="nominal",
    )
print "\tnominal event weight:", event_weight_nominal

# EXAMPLE (4): the systematic up/down event weight 
print "Example (4): the systematic up/down event weight"
for syst in ["JES", "LF", "HF", "LFStats1", "LFStats2", "HFStats1", "HFStats2", "cErr1", "cErr2"]:
    for sdir in ["Up", "Down"]:
        event_weight_shift = bweightcalc.calcEventWeight(
            jets, kind="final", systematic=syst+sdir
            )
        print "\tsystematic:", syst+sdir, ":", event_weight_shift
