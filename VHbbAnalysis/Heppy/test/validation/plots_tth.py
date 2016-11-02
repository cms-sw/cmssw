import ROOT
ROOT.gROOT.SetBatch(True)

dirs = ["Loop"]

vars_to_plot = [
    "Jet_pt",
    "Jet_eta",
    "Jet_btagCSV",
    "Jet_btagCMVA",
    "Jet_btagCMVAV2",
    "Jet_mcFlavour",
    "Jet_hadronFlavour",
    "nGenJet",
    "GenJet_pt",
    "GenJet_numBHadrons",
    "GenJet_numCHadrons",
    
    "btagWeightCSV",
    "btagWeightCSV_down_cferr1",
    "btagWeightCSV_down_cferr2",
    "btagWeightCSV_down_hf",
    "btagWeightCSV_down_hfstats1",
    "btagWeightCSV_down_hfstats2",
    "btagWeightCSV_down_jes",
    "btagWeightCSV_down_lf",
    "btagWeightCSV_down_lfstats1",
    "btagWeightCSV_down_lfstats2",
    "btagWeightCSV_up_cferr1",
    "btagWeightCSV_up_cferr2",
    "btagWeightCSV_up_hf",
    "btagWeightCSV_up_hfstats1",
    "btagWeightCSV_up_hfstats2",
    "btagWeightCSV_up_jes",
    "btagWeightCSV_up_lf",
    "btagWeightCSV_up_lfstats1",
    "btagWeightCSV_up_lfstats2",


    "btagWeightCMVAV2",
    "btagWeightCMVAV2_down_cferr1",
    "btagWeightCMVAV2_down_cferr2",
    "btagWeightCMVAV2_down_hf",
    "btagWeightCMVAV2_down_hfstats1",
    "btagWeightCMVAV2_down_hfstats2",
    "btagWeightCMVAV2_down_jes",
    "btagWeightCMVAV2_down_lf",
    "btagWeightCMVAV2_down_lfstats1",
    "btagWeightCMVAV2_down_lfstats2",
    "btagWeightCMVAV2_up_cferr1",
    "btagWeightCMVAV2_up_cferr2",
    "btagWeightCMVAV2_up_hf",
    "btagWeightCMVAV2_up_hfstats1",
    "btagWeightCMVAV2_up_hfstats2",
    "btagWeightCMVAV2_up_jes",
    "btagWeightCMVAV2_up_lf",
    "btagWeightCMVAV2_up_lfstats1",
    "btagWeightCMVAV2_up_lfstats2",

    "HLT_ttH_DL_elel",
    "HLT_ttH_DL_elmu",
    "HLT_ttH_DL_mumu",
    "HLT_ttH_SL_el",
    "HLT_ttH_SL_mu",
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
