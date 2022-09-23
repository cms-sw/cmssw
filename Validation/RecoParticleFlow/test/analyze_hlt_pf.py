#
# Usage example:
# python3 analyze_hlt_pf.py
#
# import ROOT in batch mode
import sys
import math
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
from ROOT import TF1, TF2, TH1, TH2, TH2F, TProfile, TAxis, TMath, TEllipse, TStyle, TFile, TColor, TSpectrum, TCanvas, TPad, TVirtualFitter, gStyle
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

from ctypes import c_uint8

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable() 

# Create histograms, etc.
ROOT.gROOT.SetStyle('Plain') # white background
H_NPV = ROOT.TH1F ("NPV","NPV",101,-0.5,100.5)
H_GenPi_Pt = ROOT.TH1F("H_GenPi_Pt","H_GenPi_Pt",50,0.,50.)
H_GenPi_Eta = ROOT.TH1F("H_GenPi_Eta","H_GenPi_Eta",100,-5.0,5.0)

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

# 
# Import CMS python class definitions such as Process, Source, and EDProducer
import FWCore.ParameterSet.Config as cms
ROOT.gSystem.Load("libDataFormatsHcalDetId.so")
#from DataFormats.HcalDetId import DetId

#
genparticles, genparLabel = Handle("std::vector<reco::GenParticle>"), "genParticles"
vertices, vertexLabel = Handle("std::vector<reco::Vertex>"), "hltFastPrimaryVertex"
pfcands, pfcandLabel = Handle("std::vector<reco::PFCandidate>"), "hltParticleFlow"
pfcluster, pfclusterLabel = Handle("std::vector<reco::PFCluster>"), "hltParticleFlowClusterHBHE"
pfrechit, pfrechitLabel = Handle("std::vector<reco::PFRecHit>"), ("hltParticleFlowRecHitHBHE","Cleaned","reHLT")
jets, jetLabel = Handle("std::vector<reco::PFJet>"), "hltAK4PFJets"
fatjets, fatjetLabel = Handle("std::vector<reco::PFJet>"), "hltAK8PFJets"

pfcandPtScore = Handle("edm::ValueMap<float>")
verticesScore = Handle("edm::ValueMap<float>")

if len(sys.argv)>1:
    output=sys.argv[1]
else:
    output="PF_HF_SinglePi_Pt20"
        
# open file (you can use 'edmFileUtil -d /store/whatever.root' to get the physical file name)
events = Events('reHLT_HLT.root')
    
for iev,event in enumerate(events):
    if iev >= 100: break 
    event.getByLabel(genparLabel, genparticles)
    #event.getByLabel(vertexLabel, vertices)
    event.getByLabel(pfcandLabel, pfcands)
    event.getByLabel(pfclusterLabel, pfcluster)
    event.getByLabel(pfrechitLabel, pfrechit)
    event.getByLabel(jetLabel, jets)
    event.getByLabel(fatjetLabel, fatjets)
    #event.getByLabel(pftrackLabel, pftracks)
    print("\nEvent: run %6d, lumi %4d, event %12d" % (event.eventAuxiliary().run(), event.eventAuxiliary().luminosityBlock(), event.eventAuxiliary().event()))

    # Gen particles
    for i,j in enumerate(genparticles.product()):  # loop over gen candidates
        print("GenParticle: run %6d, event %10d, genpars: pt %5.1f eta %5.2f phi %5.2f " % (event.eventAuxiliary().run(), event.eventAuxiliary().event(), j.pt(), j.eta(), j.phi() ))

    # PF clusters    
    for i,j in enumerate(pfcluster.product()):  # loop over pf clusters
        pfc = j
        fracs = pfc.recHitFractions()
        hfracs = pfc.hitsAndFractions()
        # layer 11:, 12:
        print("Cluster: run %6d, event %10d, pfclus: pt %10.5f eta %10.5f phi %10.5f layer %3d size %3d" % (event.eventAuxiliary().run(), event.eventAuxiliary().event(), pfc.pt(), pfc.eta(), pfc.phi(), pfc.layer(), fracs.size() ))
        # for e in range(0, fracs.size()):  # loop over rechits
        #     #print "  ", fracs[e], hfracs[e], hfracs[e].first, hfracs[e].second
        #     id = hfracs[e].first.rawId()
        #     print("  ", id, pfc.recHitFractions()[e].recHitRef().detId())
        #     #print HcalDetId(id).ieta(),HcalDetId(id).iphi(),HcalDetId(id).depth()
        #     # extracting ieta,iphi from detid: not working

    # PF  rechits    
    for i,j in enumerate(pfrechit.product()):  # loop over pf rechits
        # print "PFRecHit: iev %3d pfrechits %3d: energy %5.1f pt %5.1f detid %15d eta %5.2f phi %5.2f layer %3d" % ( iev, i, j.energy(), math.sqrt(j.pt2()), j.detId(), j.positionREP().eta(), j.positionREP().eta(), j.layer() )
        # access to HcalDetId and its geometrical position: not working.
        print("PFRecHit: run %6d, event %10d, pfrechits: energy %10.5f detid %15d layer %3d" % (event.eventAuxiliary().run(), event.eventAuxiliary().event(), j.energy(), j.detId(), j.layer() ))

    # AK4 jets
    for i,j in enumerate(jets.product()):  # loop over gen candidates
        print("Jet: run %6d, event %10d, jet: pt %10.5f eta %10.5f phi %10.5f " % (event.eventAuxiliary().run(), event.eventAuxiliary().event(), j.pt(), j.eta(), j.phi() ))
                
    # AK8 jets
    for i,j in enumerate(fatjets.product()):  # loop over gen candidates
        print("FatJet: run %6d, event %10d, jet: pt %10.5f eta %10.5f phi %10.5f " % (event.eventAuxiliary().run(), event.eventAuxiliary().event(), j.pt(), j.eta(), j.phi() ))
                
# Set up canvas : 
# w = 1400 
# h =  700
# can  = ROOT.TCanvas("can", "histograms   ", w, h)

#####
# ROOT.gPad.SetLogy()
# ROOT.gPad.SetLogy(0)
# H_NPV.Draw()
# can.SaveAs(output+"_NPV.pdf")
# can.SaveAs(output+"_NPV.png")
# can.SaveAs(output+"_NPV.root")

