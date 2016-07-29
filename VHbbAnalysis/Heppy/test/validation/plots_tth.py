import ROOT
ROOT.gROOT.SetBatch(True)

dirs = ["Loop_validation_tth_sl_dl_tth_hbb"]

vars_to_plot = [
    "Jet_pt",
    "Jet_eta",
    "Jet_btagCSV",
    "Jet_btagBDT",
    "Jet_mcFlavour",
    "Jet_bTagWeight",
    "Jet_bTagWeightJESUp",
    "Jet_bTagWeightJESDown",
    "Jet_bTagWeightHFUp",
    "Jet_bTagWeightHFDown",
    "Jet_bTagWeightLFUp",
    "Jet_bTagWeightLFDown",
    "Jet_bTagWeightStats1Up",
    "Jet_bTagWeightStats1Down",
    "Jet_bTagWeightStats2Up",
    "Jet_bTagWeightStats2Down",
    "nGenJet",
    "GenJet_pt",
    "GenJet_numBHadrons",
    "GenJet_numCHadrons",
]

def process_dir(d):
    print "Processing",d
    tf = ROOT.TFile(d + "/tree.root")
    tt = tf.Get("tree")
    if tt.GetEntries() <= 100:
        print "WARN: low efficiency", d
   
    npos = tf.Get("CountPosWeight").GetBinContent(1)
    nneg = tf.Get("CountNegWeight").GetBinContent(1)
    ntot = npos + nneg
    print "Ngen", ntot, npos, nneg

    for v in vars_to_plot:
        tt.Draw(v + " >> h")
        h = tf.Get("h")
        print v, round(h.Integral(), 2), round(h.GetMean(), 2), round(h.GetRMS(), 2) 

for d in dirs:
    process_dir(d)
