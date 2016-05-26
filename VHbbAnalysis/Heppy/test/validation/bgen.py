import ROOT
import os
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2(True)
import numpy as n  

out = ROOT.TFile("BFilter.root", "UPDATE")

#path = "dcap://t3se01.psi.ch:22125///pnfs/psi.ch/cms/trivcat/store/t3groups/ethz-higgs/run2/VHBBHeppyV14/"
#path = "root://stormgf1.pi.infn.it:1094//store/user/arizzi/VHBBHeppyV14/"

samples = {

    "Z_inclusive" : {
        "name" : "dcap://t3se01.psi.ch:22125///pnfs/psi.ch/cms/trivcat/store/t3groups/ethz-higgs/run2/VHBBHeppyV14/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/VHBB_HEPPY_V14_DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8__RunIISpring15MiniAODv2-74X_mcRun2_asymptotic_v2-v1/151025_092939/0000/",
        "xsec" : 6025.2/1.23,
        "files" : 108,
        "tree" : ROOT.TTree("tree_Z_inclusive", "tree title"),
        "weight" : 1.0
        },

    "Z_BJets" : {
        "name" : "dcap://t3se01.psi.ch:22125///pnfs/psi.ch/cms/trivcat/store/t3groups/ethz-higgs/run2/VHBBHeppyV14/VHBB_HEPPY_V14_DYBJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8__RunIISpring15MiniAODv2-74X_mcRun2_asymptotic_v2-v1/151113_073851/0000/",
        "xsec" : 71.77,    
        "files" : 27 ,
        "tree" : ROOT.TTree("tree_Z_BJets", "tree title"),
        "weight" : 1.0
        },

    "Z_BGenFilter" : {
        "name" : "dcap://t3se01.psi.ch:22125///pnfs/psi.ch/cms/trivcat/store/t3groups/ethz-higgs/run2/VHBBHeppyV14/DYJetsToLL_BGenFilter_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/VHBB_HEPPY_V14_DYJetsToLL_BGenFilter_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8__RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/151118_082615/  ",
        "xsec" : 228.9,
        "tree" : ROOT.TTree("tree_Z_BGenFilter", "tree title"),
        "weight" : 1.0
        },

    "W_inclusive" : {
        "name" : "dcap://t3se01.psi.ch:22125///pnfs/psi.ch/cms/trivcat/store/t3groups/ethz-higgs/run2/VHBBHeppyV14//WJetsToLNu_TuneCUETP8M1_13TeV-madgraphMLM-pythia8//VHBB_HEPPY_V14_WJetsToLNu_TuneCUETP8M1_13TeV-madgraphMLM-pythia8__RunIISpring15MiniAODv2-74X_mcRun2_asymptotic_v2-v1//151024_220106/0000/",
        "xsec" : 61526.7/1.23,
        "files" : 200,#811,
        "tree" : ROOT.TTree("tree_W_inclusive", "tree title"),
        "weight" : 1.0
        },

    "W_BJets" : {
        "name" : "root://stormgf1.pi.infn.it:1094//store/user/arizzi/VHBBHeppyV14//WBJetsToLNu_Wpt-40toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/VHBB_HEPPY_V14_WBJetsToLNu_Wpt-40toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8__RunIISpring15MiniAODv2-74X_mcRun2_asymptotic_v2-v1/151113_074533/0000/",
        "xsec" : 34.2,
        "files" : 24,
        "tree" : ROOT.TTree("tree_W_BJets", "tree title"),
        "weight" : 1.0
        },

    "W_BGenFilter" : {
        "name" : "root://stormgf1.pi.infn.it:1094//store/user/arizzi/VHBBHeppyV14/WJetsToLNu_BGenFilter_Wpt-40toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/VHBB_HEPPY_V14_WJetsToLNu_BGenFilter_Wpt-40toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8__RunIISpring15MiniAODv2-Asympt25ns_74X_mcRun2_asymptotic_v2-v1/160210_090632/0000/",
        "xsec" : 203.8,
        "files" : 81,
        "tree" : ROOT.TTree("tree_W_BGenFilter", "tree title"),
        "weight" : 1.0
        },
}


weight_ = n.zeros(1, dtype=float)
ttCls_ = n.zeros(1, dtype=float)
nGenStatus2bHad_ = n.zeros(1, dtype=float)
lheNb_  = n.zeros(1, dtype=float) 
lheVpt_  = n.zeros(1, dtype=float) 

for it in samples.keys():
    samples[it]["tree"].Branch('weight', weight_, 'weight/D')
    samples[it]["tree"].Branch('ttCls', ttCls_, 'ttCls/D')
    samples[it]["tree"].Branch('nGenStatus2bHad', nGenStatus2bHad_, 'nGenStatus2bHad/D')
    samples[it]["tree"].Branch('lheNb', lheNb_, 'lheNb/D')
    samples[it]["tree"].Branch('lheV_pt', lheVpt_, 'lheV_pt/D')

for run in [
    "W_inclusive", 
    "W_BJets", 
    "W_BGenFilter"
    ]:
    
    print "Now doing sample.......", run

    file_names = []
    for p in range(1, samples[run]["files"] ):
        file_names.append( "tree_"+str(p)+".root" )
    print "Added ", len(file_names), "files"

    chain = ROOT.TChain("tree")
    CountWeighted = 0
    for file_name in file_names:
        f = ROOT.TFile.Open( samples[run]["name"]  + "/" + file_name )
        if f==None or f.IsZombie():
            continue
        f.cd()
        count = f.Get("CountWeighted")
        CountWeighted += count.GetBinContent(1)
        f.Close()
        chain.AddFile( samples[run]["name"] + "/" + file_name )

    samples[run]["weight"] = 10./(CountWeighted/samples[run]["xsec"])*1000
    
    print "Total processed events: ", CountWeighted
    print "Cross section: ", samples[run]["xsec"], "pb"
    print "Weight to 10fb-1: ", samples[run]["weight"] 
            
    chain.SetBranchStatus("*",        False)
    #chain.SetBranchStatus("nJet",      True)
    #chain.SetBranchStatus("Jet_*",     True) 
    #chain.SetBranchStatus("nGenJet",   True)
    #chain.SetBranchStatus("GenJet_*",  True)
    #chain.SetBranchStatus("puWeight",  True)
    #chain.SetBranchStatus("genWeight", True)
    chain.SetBranchStatus("ttCls", True)
    chain.SetBranchStatus("nGenStatus2bHad", True)
    chain.SetBranchStatus("lheNb", True)
    chain.SetBranchStatus("lheV_pt", True)

    print chain.GetEntries()
    for iev in range( min(1e+11, chain.GetEntries()) ):

        chain.GetEntry(iev)
        ev = chain
        if iev%10000 == 0:
            print "Processing event ", iev
        weight_[0] = samples[run]["weight"] 
        ttCls_[0] = ev.ttCls
        nGenStatus2bHad_[0] = ev.nGenStatus2bHad
        lheNb_[0] = ev.lheNb
        lheVpt_[0] = ev.lheV_pt
        samples[run]["tree"].Fill()

    out.cd()
    samples[run]["tree"].Write("", ROOT.TObject.kOverwrite)

out.Close()
