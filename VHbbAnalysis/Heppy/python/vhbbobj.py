
#!/bin/env python
from math import *
import ROOT
#from CMGTools.TTHAnalysis.signedSip import *
from PhysicsTools.Heppy.analyzers.objects.autophobj import *
from PhysicsTools.HeppyCore.utils.deltar import deltaPhi, deltaR
from VHbbAnalysis.Heppy.signedSip import qualityTrk
from VHbbAnalysis.Heppy.leptonMVA import ptRelv2

import copy, os

leptonTypeVHbb = NTupleObjectType("leptonTypeVHbb", baseObjectTypes = [ leptonType ], variables = [
    # Loose id 
    NTupleVariable("looseIdSusy", lambda x : x.looseIdSusy if hasattr(x, 'looseIdSusy') else -1, int, help="Loose ID for Susy ntuples (always true on selected leptons)"),
    NTupleVariable("looseIdPOG", lambda x : x.muonID("POG_ID_Loose") if abs(x.pdgId()) == 13 else -1, int, help="Loose ID for Susy ntuples (always true on selected leptons)"),
    # Isolations with the two radia
    NTupleVariable("chargedHadRelIso03",  lambda x : x.chargedHadronIsoR(0.3)/x.pt(), help="PF Rel Iso, R=0.3, charged hadrons only"),
    NTupleVariable("chargedHadRelIso04",  lambda x : x.chargedHadronIsoR(0.4)/x.pt(), help="PF Rel Iso, R=0.4, charged hadrons only"),
    NTupleVariable("eleSieie",    lambda x : x.full5x5_sigmaIetaIeta() if abs(x.pdgId())==11 else -1., help="sigma IEtaIEta for electrons"),
    NTupleVariable("eleDEta",     lambda x : x.deltaEtaSuperClusterTrackAtVtx() if abs(x.pdgId())==11 else -1., help="delta eta for electrons"),
    NTupleVariable("eleDPhi",     lambda x : x.deltaPhiSuperClusterTrackAtVtx() if abs(x.pdgId())==11 else -1., help="delta phi for electrons"),
    NTupleVariable("eleHoE",      lambda x : x.hadronicOverEm() if abs(x.pdgId())==11 else -1., help="H/E for electrons"),
    NTupleVariable("eleMissingHits",      lambda x : x.lostInner() if abs(x.pdgId())==11 else -1., help="Missing hits for electrons"),
    NTupleVariable("eleChi2",      lambda x : x.gsfTrack().normalizedChi2() if abs(x.pdgId())==11 else -1., help="Track chi squared for electrons' gsf tracks"),
    # Extra electron id variables
    NTupleVariable("convVetoFull", lambda x : (x.passConversionVeto() and x.lostInner() == 0) if abs(x.pdgId())==11 else 1, int, help="Conv veto + no missing hits for electrons, always true for muons."),
    #NTupleVariable("eleMVArawPhys14NonTrig", lambda x : x.mvaRun2("NonTrigPhys14") if abs(x.pdgId()) == 11 else -1, help="EGamma POG MVA ID for non-triggering electrons (raw MVA value, Phys14 training); 1 for muons"),
    #NTupleVariable("eleMVAIdPhys14NonTrig", lambda x : max(x.electronID("POG_MVA_ID_Phys14_NonTrig_VLoose"), 2*x.electronID("POG_MVA_ID_Phys14_NonTrig_Loose"), 3*x.electronID("POG_MVA_ID_Phys14_NonTrig_Tight")) if abs(x.pdgId()) == 11 else -1, int, help="EGamma POG MVA ID for non-triggering electrons (0=none, 1=vloose, 2=loose, 3=tight, Phys14 training); 1 for muons"),
    NTupleVariable("eleMVArawSpring15Trig", lambda x : getattr(x,"mvaRawSpring15Trig",-2) if abs(x.pdgId()) == 11 else -1, help="EGamma POG MVA ID for triggering electrons (raw MVA value, Spring15 training); 1 for muons"),
    NTupleVariable("eleMVAIdSpring15Trig", lambda x : max(x.mvaIdSpring15TrigMedium, 2*x.mvaIdSpring15TrigTight) if abs(x.pdgId()) == 11 and hasattr(x,"mvaIdSpring15TrigMedium") else -1, int, help="EGamma POG MVA ID for triggering electrons (0=none, 1=WP90, 2=WP80, Spring15 training); 1 for muons"),
    NTupleVariable("eleMVArawSpring15NonTrig", lambda x : getattr(x,"mvaRawSpring15NonTrig",-2) if abs(x.pdgId()) == 11 else -1, help="EGamma POG MVA ID for non-triggering electrons (raw MVA value, Spring15 training); 1 for muons"),
    NTupleVariable("eleMVAIdSpring15NonTrig", lambda x : max(x.mvaIdSpring15NonTrigMedium, 2*x.mvaIdSpring15NonTrigTight) if abs(x.pdgId()) == 11 and hasattr(x,"mvaIdSpring15NonTrigMedium")  else -1, int, help="EGamma POG MVA ID for non-triggering electrons (0=none, 1=WP90, 2=WP80, Spring15 training); 1 for muons"),
    ##NTupleVariable("tightCharge",  lambda lepton : ( lepton.isGsfCtfScPixChargeConsistent() + lepton.isGsfScPixChargeConsistent() ) if abs(lepton.pdgId()) == 11 else 2*(lepton.innerTrack().ptError()/lepton.innerTrack().pt() < 0.2), int, help="Tight charge criteria"),
    # Muon-speficic info
    NTupleVariable("nStations",    lambda lepton : lepton.numberOfMatchedStations() if abs(lepton.pdgId()) == 13 else 4, help="Number of matched muons stations (4 for electrons)"),
    NTupleVariable("trkKink",      lambda lepton : lepton.combinedQuality().trkKink if abs(lepton.pdgId()) == 13 else 0, help="Tracker kink-finder"),
    NTupleVariable("segmentCompatibility",      lambda lepton : lepton.segmentCompatibility() if abs(lepton.pdgId()) == 13 else 0, help="Segment compatibility"), 
    NTupleVariable("caloCompatibility",      lambda lepton : lepton.caloCompatibility() if abs(lepton.pdgId()) == 13 else 0, help="Calorimetric compatibility"), 
    NTupleVariable("globalTrackChi2",      lambda lepton : lepton.globalTrack().normalizedChi2() if abs(lepton.pdgId()) == 13 and lepton.globalTrack().isNonnull() else 0, help="Global track normalized chi2"), 
    NTupleVariable("nChamberHits", lambda lepton: lepton.globalTrack().hitPattern().numberOfValidMuonHits() if abs(lepton.pdgId()) == 13 and lepton.globalTrack().isNonnull() else -1, help="Number of muon chamber hits (-1 for electrons)"),
    NTupleVariable("isPFMuon", lambda lepton: lepton.isPFMuon() if abs(lepton.pdgId()) == 13 else 0, help="1 if muon passes particle flow ID"),
    NTupleVariable("isGlobalMuon", lambda lepton: lepton.isGlobalMuon() if abs(lepton.pdgId()) == 13 else 0, help="1 if muon is global muon"),
    NTupleVariable("isTrackerMuon", lambda lepton: lepton.isTrackerMuon() if abs(lepton.pdgId()) == 13 else 0, help="1 if muon is tracker muon"),
    NTupleVariable("pixelHits", lambda lepton : lepton.innerTrack().hitPattern().numberOfValidPixelHits() if abs(lepton.pdgId()) == 13 and lepton.innerTrack().isNonnull() else -1, help="Number of pixel hits (-1 for electrons)"),
    # Extra tracker-related id variables
    NTupleVariable("trackerLayers", lambda x : (x.track() if abs(x.pdgId())==13 else x.gsfTrack()).hitPattern().trackerLayersWithMeasurement(), int, help="Tracker Layers"),
    NTupleVariable("pixelLayers", lambda x : (x.track() if abs(x.pdgId())==13 else x.gsfTrack()).hitPattern().pixelLayersWithMeasurement(), int, help="Pixel Layers"),
    # TTH-id related variables
    NTupleVariable("mvaTTH",     lambda lepton : lepton.mvaValueTTH if hasattr(lepton,'mvaValueTTH') else -1, help="Lepton MVA (ttH version)"),
    NTupleVariable("jetOverlapIdx", lambda lepton : getattr(lepton, "jetOverlapIdx", -1), int, help="index of jet with overlapping PF constituents. If idx>=1000, then idx = idx-1000 and refers to discarded jets."),
    NTupleVariable("jetPtRatio", lambda lepton : lepton.pt()/lepton.jet.pt() if hasattr(lepton,'jet') else -1, help="pt(lepton)/pt(nearest jet)"),
    NTupleVariable("jetBTagCSV", lambda lepton : lepton.jet.btag('pfCombinedInclusiveSecondaryVertexV2BJetTags') if hasattr(lepton,'jet') and hasattr(lepton.jet, 'btag') else -99, help="btag of nearest jet"),
    NTupleVariable("jetDR",      lambda lepton : deltaR(lepton.eta(),lepton.phi(),lepton.jet.eta(),lepton.jet.phi()) if hasattr(lepton,'jet') else -1, help="deltaR(lepton, nearest jet)"),
    NTupleVariable("mvaTTHjetPtRatio", lambda lepton : lepton.pt()/lepton.jet_leptonMVA.pt() if hasattr(lepton,'jet_leptonMVA') else -1, help="pt(lepton)/pt(nearest jet with pT > 25 GeV)"),
    NTupleVariable("mvaTTHjetBTagCSV", lambda lepton : lepton.jet_leptonMVA.btag('combinedInclusiveSecondaryVertexV2BJetTags') if hasattr(lepton,'jet_leptonMVA') and hasattr(lepton.jet_leptonMVA, 'btag') else -99, help="btag of nearest jet with pT > 25 GeV"),    
    NTupleVariable("mvaTTHjetDR",      lambda lepton : deltaR(lepton.eta(),lepton.phi(),lepton.jet_leptonMVA.eta(),lepton.jet_leptonMVA.phi()) if hasattr(lepton,'jet_leptonMVA') else -1, help="deltaR(lepton, nearest jet with pT > 25 GeV)"),
    NTupleVariable("pfRelIso03",      lambda ele : (ele.pfIsolationVariables().sumChargedHadronPt + max(ele.pfIsolationVariables().sumNeutralHadronEt + ele.pfIsolationVariables().sumPhotonEt - 0.5 * ele.pfIsolationVariables().sumPUPt,0.0)) / ele.pt()  if abs(ele.pdgId()) == 11 else -1, help="0.3 particle based iso"),
    NTupleVariable("pfRelIso04",      lambda mu : (mu.pfIsolationR04().sumChargedHadronPt + max( mu.pfIsolationR04().sumNeutralHadronEt + mu.pfIsolationR04().sumPhotonEt - 0.5 * mu.pfIsolationR04().sumPUPt,0.0)) / mu.pt() if abs(mu.pdgId()) == 13 else -1, help="0.4 particle based iso"),
    NTupleVariable("etaSc", lambda x : x.superCluster().eta() if abs(x.pdgId())==11 else -100, help="Electron supercluster pseudorapidity"),
    NTupleVariable("eleExpMissingInnerHits", lambda x : x.gsfTrack().hitPattern().numberOfHits(ROOT.reco.HitPattern.MISSING_INNER_HITS) if abs(x.pdgId())==11 else -1, help="Electron expected missing inner hits"),
    NTupleVariable("eleooEmooP", lambda x : abs(1.0/x.ecalEnergy() - x.eSuperClusterOverP()/x.ecalEnergy()) if abs(x.pdgId())==11 and x.ecalEnergy()>0.0 else 9e9 , help="Electron 1/E - 1/P"),
    NTupleVariable("dr03TkSumPt", lambda x : x.dr03TkSumPt() if abs(x.pdgId())==11 else 0.0 , help="Electron track sum pt"),
    NTupleVariable("eleEcalClusterIso", lambda x : x.ecalPFClusterIso() if abs(x.pdgId())==11 else 0.0 , help="Electron ecal cluster iso"),
    NTupleVariable("eleHcalClusterIso", lambda x : x.hcalPFClusterIso() if abs(x.pdgId())==11 else 0.0 , help="Electron hcal cluster iso"),
    NTupleVariable("miniIsoCharged", lambda x : x.miniAbsIsoCharged if hasattr(x,'miniAbsIsoCharged') else  -999, help="PF miniIso (charged) in GeV"),
    NTupleVariable("miniIsoNeutral", lambda x : x.miniAbsIsoNeutral if hasattr(x,'miniAbsIsoNeutral') else  -999, help="PF miniIso (neutral) in GeV"),
    NTupleVariable("mvaTTHjetPtRel", lambda x : ptRelv2(x) if hasattr(x,'jet') else -1, help="jetPtRel variable used by ttH multilepton MVA"),
    NTupleVariable("mvaTTHjetNDauChargedMVASel", lambda lepton: sum((deltaR(x.eta(),x.phi(),lepton.jet.eta(),lepton.jet.phi())<=0.4 and x.charge()!=0 and x.fromPV()>1 and qualityTrk(x.pseudoTrack(),lepton.associatedVertex)) for x in lepton.jet.daughterPtrVector()) if hasattr(lepton,'jet') and lepton.jet != lepton else 0, help="jetNDauChargedMVASel variable used by ttH multilepton MVA"),
    NTupleVariable("uncalibratedPt", lambda x : getattr(x,"uncalibratedP4").Pt() if abs(x.pdgId())==11 and hasattr(x,"uncalibratedP4") else x.pt() , help="Electron uncalibrated pt"),
    # MC-match info
#    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (25 for H, 6 for t, 23/24 for W/Z)"),
#    NTupleVariable("mcMatchAny",  lambda x : x.mcMatchAny, int, mcOnly=True, help="Match to any final state leptons: -mcMatchId if prompt, 0 if unmatched, 1 if light flavour, 2 if heavy flavour (b)"),
#    NTupleVariable("mcMatchTau",  lambda x : x.mcMatchTau, int, mcOnly=True, help="True if the leptons comes from a tau"),
])

##------------------------------------------  
## TAU
##------------------------------------------  

tauTypeVHbb = NTupleObjectType("tauTypeVHbb", baseObjectTypes = [ tauType ], variables = [
    NTupleVariable("idxJetMatch", lambda x : x.jetIdx, int, help="index of the matching jet"),
    NTupleVariable("genMatchType", lambda x : x.genMatchType, int,mcOnly=True, help="..FILLME PLEASE..")
])

##------------------------------------------  
## JET
##------------------------------------------  

jetTypeVHbb = NTupleObjectType("jet",  baseObjectTypes = [ jetType ], variables = [
    NTupleVariable("rawPtAfterSmearing",  lambda x : x.pt() / getattr(x, 'corr', 1) , help="p_{T} before JEC but including JER effect"),
    NTupleVariable("idxFirstTauMatch", lambda x : x.tauIdxs[0] if len(getattr(x, "tauIdxs", [])) > 0 else -1, int,help='index of the first matching tau'),
    NTupleVariable("heppyFlavour", lambda x : x.mcFlavour, int,     mcOnly=True, help="heppy-style match to gen quarks"),
#    NTupleVariable("hadronFlavour", lambda x : x.hadronFlavour(), int,     mcOnly=True, help="hadron flavour (ghost matching to B/C hadrons)"),
    NTupleVariable("ctagVsL", lambda x : x.bDiscriminator('pfCombinedCvsLJetTags'), help="c-btag vs light jets"),
    NTupleVariable("ctagVsB", lambda x : x.bDiscriminator('pfCombinedCvsBJetTags'), help="c-btag vs light jets"),
    NTupleVariable("btagBDT", lambda x : getattr(x,"btagBDT",-99), help="combined super-btag"),
    NTupleVariable("btagProb", lambda x : x.btag('pfJetProbabilityBJetTags') , help="jet probability b-tag"),
    NTupleVariable("btagBProb", lambda x : x.btag('pfJetBProbabilityBJetTags') , help="jet b-probability b-tag"),
    NTupleVariable("btagSoftEl", lambda x : getattr(x, "btagSoftEl", -1000) , help="soft electron b-tag"),
    NTupleVariable("btagSoftMu", lambda x : getattr(x, "btagSoftMu", -1000) , help="soft muon b-tag"),
    NTupleVariable("btagHip",   lambda x : x.bDiscriminator("newpfCombinedInclusiveSecondaryVertexV2BJetTags"), help="pfCombinedInclusiveSVV2 with btv HIP mitigation"),
    NTupleVariable("btagHip2",   lambda x : getattr(x,"btagHip",-2), help="pfCombinedInclusiveSVV2 with btv HIP mitigation"),
    NTupleVariable("btagHipCMVA",   lambda x : getattr(x,"btagHip",-2), help="CMVAV2 with btv HIP mitigation"),
    NTupleVariable("btagCSVV0",   lambda x : x.bDiscriminator('pfCombinedSecondaryVertexV2BJetTags'), help="should be the old CSV discriminator with AVR vertices"),
    NTupleVariable("btagCMVAV2",  lambda x : x.btag('pfCombinedMVAV2BJetTags'), help="CMVA V2 discriminator"),
   # NTupleVariable("mcMatchId",    lambda x : x.mcMatchId,   int, mcOnly=True, help="Match to source from hard scatter (25 for H, 6 for t, 23/24 for W/Z)"),
   # NTupleVariable("puId", lambda x : x.puJetIdPassed, int,     mcOnly=False, help="puId (full MVA, loose WP, 5.3.X training on AK5PFchs: the only thing that is available now)"),
   # NTupleVariable("id",    lambda x : x.jetID("POG_PFID") , int, mcOnly=False,help="POG Loose jet ID"),
    NTupleVariable("chHEF", lambda x : x.chargedHadronEnergy()/(x.p4()*x.rawFactor()).energy(), float, mcOnly = False, help="chargedHadronEnergyFraction (relative to uncorrected jet energy)"),
    NTupleVariable("neHEF", lambda x : x.neutralHadronEnergy()/(x.p4()*x.rawFactor()).energy(), float, mcOnly = False,help="neutralHadronEnergyFraction (relative to uncorrected jet energy)"),
    NTupleVariable("chEmEF", lambda x : x.chargedEmEnergy()/(x.p4()*x.rawFactor()).energy(), float, mcOnly = False,help="chargedEmEnergyFraction (relative to uncorrected jet energy)"),
    NTupleVariable("neEmEF", lambda x : x.neutralEmEnergy()/(x.p4()*x.rawFactor()).energy(), float, mcOnly = False,help="neutralEmEnergyFraction (relative to uncorrected jet energy)"),
    NTupleVariable("muEF", lambda x : x.muonEnergy()/(x.p4()*x.rawFactor()).energy(), float, mcOnly = False,help="muon energy fraction (relative to uncorrected jet energy)"),
    NTupleVariable("chMult", lambda x : x.chargedMultiplicity(), int, mcOnly = False,help="chargedMultiplicity from PFJet.h"),
    NTupleVariable("nhMult", lambda x : x.neutralMultiplicity(), int, mcOnly = False,help="neutralMultiplicity from PFJet.h"),
    NTupleVariable("leadTrackPt", lambda x : x.leadTrackPt() , float, mcOnly = False, help="pt of the leading track in the jet"), 
    NTupleVariable("mcEta",   lambda x : x.mcJet.eta() if getattr(x,"mcJet",None) else 0., mcOnly=True, help="eta of associated gen jet"),
    NTupleVariable("mcPhi",   lambda x : x.mcJet.phi() if getattr(x,"mcJet",None) else 0., mcOnly=True, help="phi of associated gen jet"),
    NTupleVariable("mcM",   lambda x : x.mcJet.p4().M() if getattr(x,"mcJet",None) else 0., mcOnly=True, help="mass of associated gen jet"),
    NTupleVariable("leptonPdgId",   lambda x : x.leptons[0].pdgId() if len(x.leptons) > 0 else -99, mcOnly=False, help="pdg id of the first associated lepton"),
    NTupleVariable("leptonPt",   lambda x : x.leptons[0].pt() if len(x.leptons) > 0 else -99, mcOnly=False, help="pt of the first associated lepton"),
    NTupleVariable("leptonPtRel",   lambda x : ptRel(x.leptons[0].p4(),x.p4()) if len(x.leptons) > 0 else -99, mcOnly=False, help="ptrel of the first associated lepton"),
    NTupleVariable("leptonPtRelInv",   lambda x : ptRel(x.p4(),x.leptons[0].p4()) if len(x.leptons) > 0 else -99, mcOnly=False, help="ptrel Run1 definition of the first associated lepton"),
    NTupleVariable("leptonDeltaR",   lambda x : deltaR(x.leptons[0].p4().eta(),x.leptons[0].p4().phi(),x.p4().eta(),x.p4().phi()) if len(x.leptons) > 0 else -99, mcOnly=False, help="deltaR of the first associated lepton"),
    NTupleVariable("leptonDeltaPhi",   lambda x : deltaPhi(x.leptons[0].p4().phi(),x.p4().phi()) if len(x.leptons) > 0 else 0, mcOnly=False, help="deltaPhi of the first associated lepton"),
    NTupleVariable("leptonDeltaEta",   lambda x : x.leptons[0].p4().eta()-x.p4().eta() if len(x.leptons) > 0 else 0, mcOnly=False, help="deltaEta of the first associated lepton"),
    NTupleVariable("vtxMass",   lambda x : x.userFloat("vtxMass"), mcOnly=False, help="vtxMass from btag"),
    NTupleVariable("vtxNtracks",   lambda x : x.userFloat("vtxNtracks"), mcOnly=False, help="number of tracks at vertex from btag"),
    NTupleVariable("vtxPt",   lambda x : sqrt(x.userFloat("vtxPx")**2 + x.userFloat("vtxPy")**2), mcOnly=False, help="pt of vertex from btag"),
    NTupleVariable("vtx3DSig",   lambda x : x.userFloat("vtx3DSig"), mcOnly=False, help="decay len significance of vertex from btag"),
    NTupleVariable("vtx3DVal",   lambda x : x.userFloat("vtx3DVal"), mcOnly=False, help="decay len of vertex from btag"),
    NTupleVariable("vtxPosX",   lambda x : x.userFloat("vtxPosX"), mcOnly=False, help="X coord of vertex from btag"),
    NTupleVariable("vtxPosY",   lambda x : x.userFloat("vtxPosY"), mcOnly=False, help="Y coord of vertex from btag"), 
    NTupleVariable("vtxPosZ",   lambda x : x.userFloat("vtxPosZ"), mcOnly=False, help="Z coord of vertex from btag"),
    NTupleVariable("pullVectorPhi", lambda x : getattr(x,"pullVectorPhi",-99), mcOnly=False, help="pull angle phi in the phi eta plane"),
    NTupleVariable("pullVectorMag", lambda x : getattr(x,"pullVectorMag",-99), mcOnly=False, help="pull angle magnitude"),
   # QG variables:
# this computes for all
#    NTupleVariable("qgl",   lambda x :x.qgl() , float, mcOnly=False,help="QG Likelihood"),
#    NTupleVariable("ptd",   lambda x : getattr(x.computeQGvars(),'ptd', 0), float, mcOnly=False,help="QG input variable: ptD"),
#    NTupleVariable("axis2",   lambda x : getattr(x.computeQGvars(),'axis2', 0) , float, mcOnly=False,help="QG input variable: axis2"),
#    NTupleVariable("mult",   lambda x : getattr(x.computeQGvars(),'mult', 0) , int, mcOnly=False,help="QG input variable: total multiplicity"),

# this only read qgl if it was explicitelly computed in the code
    NTupleVariable("qgl",   lambda x : getattr(x,'qgl_value',-20) , float, mcOnly=False,help="QG Likelihood"),
    NTupleVariable("ptd",   lambda x : getattr(x,'ptd', -20), float, mcOnly=False,help="QG input variable: ptD"),
    NTupleVariable("axis2",   lambda x : getattr(x,'axis2', -20) , float, mcOnly=False,help="QG input variable: axis2"),
    NTupleVariable("mult",   lambda x : getattr(x,'mult', -20) , int, mcOnly=False,help="QG input variable: total multiplicity"),
    NTupleVariable("numberOfDaughters",   lambda x : x.numberOfDaughters(), int, mcOnly=False,help="number of daughters"),
    NTupleVariable("btagIdx",   lambda x : x.btagIdx, int, mcOnly=False,help="ranking in btag"),
    NTupleVariable("mcIdx",   lambda x : x.mcJet.index if hasattr(x,"mcJet") and x.mcJet is not None else -1, int, mcOnly=False,help="index of the matching gen jet"),
    #NTupleVariable("pt_reg",lambda x : getattr(x,"pt_reg",-99), help="Regression"),
    #NTupleVariable("pt_regVBF",lambda x : getattr(x,"pt_regVBF",-99), help="Regression for VBF"),
    NTupleVariable("blike_VBF",lambda x : getattr(x,"blike_VBF",-2), help="VBF blikelihood for SingleBtag dataset")
 ])

# "" is the nominal rgression, the other refer to JEC/JER up/down
for analysis in ["","corrJECUp", "corrJECDown", "corrJERUp", "corrJERDown"]:
    jetTypeVHbb.variables += [NTupleVariable("pt_reg"+("_"+analysis if analysis!="" else ""), lambda x, analysis=analysis : getattr(x,"pt_reg"+analysis,-99), help="Regression "+analysis)]
    jetTypeVHbb.variables += [NTupleVariable("pt_regVBF"+("_"+analysis if analysis!="" else ""), lambda x, analysis=analysis : getattr(x,"pt_regVBF"+analysis,-99), help="Regressionfor VBF "+analysis)]


#add per-jet b-tag systematic weight
'''
from PhysicsTools.Heppy.physicsutils.BTagWeightCalculator import BTagWeightCalculator
csvpath = os.environ['CMSSW_BASE']+"/src/VHbbAnalysis/Heppy/data/csv"
bweightcalc = BTagWeightCalculator(
    csvpath + "/csv_rwt_fit_hf_76x_2016_02_08.root", 
    csvpath + "/csv_rwt_fit_lf_76x_2016_02_08.root", 
)

for syst in ["JES", "LF", "HF", "HFStats1", "HFStats2", "LFStats1", "LFStats2", "cErr1", "cErr2"]:
    for sdir in ["Up", "Down"]:
        jetTypeVHbb.variables += [NTupleVariable("bTagWeight"+syst+sdir,
            lambda jet, sname=syst+sdir,bweightcalc=bweightcalc: bweightcalc.calcJetWeight(
                jet, kind="final", systematic=sname
            ), float, mcOnly=True, help="b-tag CSV weight, variating "+syst + " "+sdir
        )]
jetTypeVHbb.variables += [NTupleVariable("bTagWeight",
    lambda jet, bweightcalc=bweightcalc: bweightcalc.calcJetWeight(
        jet, kind="final", systematic="nominal",
    ), float, mcOnly=True, help="b-tag CSV weight, nominal"
)]
'''

# add the POG SF
from btagSF import *

for algo in ["CSV", "CMVAV2"]:
    for wp in [ "L", "M", "T" ]:
        for syst in ["central", "up", "down"]:
            syst_name = "" if syst=="central" else ("_"+syst) 
            jetTypeVHbb.variables += [ NTupleVariable("btag"+algo+wp+"_SF"+syst_name,  lambda x, get_SF=get_SF, syst=syst, algo=algo, wp=wp, btag_calibrators=btag_calibrators : 
                                                      get_SF(x.pt(), x.eta(), x.hadronFlavour(), 0.0, syst, algo, wp, False, btag_calibrators)
                                                      , float, mcOnly=True, help="b-tag "+algo+wp+" POG scale factor, "+syst  )]

    for syst in ["central", "up_jes", "down_jes", "up_lf", "down_lf", "up_hf", "down_hf", "up_hfstats1", "down_hfstats1", "up_hfstats2", "down_hfstats2", "up_lfstats1", "down_lfstats1", "up_lfstats2", "down_lfstats2", "up_cferr1", "down_cferr1", "up_cferr2", "down_cferr2"]:
        syst_name = "" if syst=="central" else ("_"+syst) 
        jetTypeVHbb.variables += [ NTupleVariable("btagWeight"+algo+syst_name,  lambda x, get_SF=get_SF, syst=syst, algo=algo, wp=wp, btag_calibrators=btag_calibrators : 
                                                      get_SF(x.pt(), x.eta(), x.hadronFlavour(), (x.btag("pfCombinedInclusiveSecondaryVertexV2BJetTags") if algo=="CSV" else x.btag('pfCombinedMVAV2BJetTags')), syst, algo, wp, True, btag_calibrators)
                                                      , float, mcOnly=True, help="b-tag "+algo+" continuous POG scale factor, "+syst  )]
        
#add per-lepton SF
from leptonSF import LeptonSF

jsonpath = os.environ['CMSSW_BASE']+"/src/VHbbAnalysis/Heppy/data/leptonSF/"
jsons = {    
    #OLD
    'muEff_HLT_RunC' : [ jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15_eff.json' , 'runC_IsoMu20_OR_IsoTkMu20_PtEtaBins', 'abseta_pt_MC' ],
    'muEff_HLT_RunD4p2' : [ jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15_eff.json' , 'runD_IsoMu20_OR_IsoTkMu20_HLTv4p2_PtEtaBins', 'abseta_pt_MC' ],
    'muEff_HLT_RunD4p3' : [ jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15_eff.json' , 'runD_IsoMu20_OR_IsoTkMu20_HLTv4p3_PtEtaBins', 'abseta_pt_MC' ],
    'muSF_HLT_RunC' : [ jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15.json' , 'runC_IsoMu20_OR_IsoTkMu20_PtEtaBins', 'abseta_pt_ratio' ],
    #'muSF_HLT_RunD4p2' : [ jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15.json' , 'runD_IsoMu20_OR_IsoTkMu20_HLTv4p2_PtEtaBins', 'abseta_pt_ratio' ],
    #'muSF_HLT_RunD4p3' : [ jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15.json' , 'runD_IsoMu20_OR_IsoTkMu20_HLTv4p3_PtEtaBins', 'abseta_pt_ratio' ],
    #'muSF_IsoLoose' : [ jsonpath+'MuonIso_Z_RunCD_Reco76X_Feb15.json' , 'MC_NUM_LooseRelIso_DEN_LooseID_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
    #'muSF_IsoTight' : [ jsonpath+'MuonIso_Z_RunCD_Reco76X_Feb15.json' , 'MC_NUM_TightRelIso_DEN_TightID_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
    #'muSF_IdCutLoose' : [ jsonpath+'MuonID_Z_RunCD_Reco76X_Feb15.json' , 'MC_NUM_LooseID_DEN_genTracks_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'] ,
    #'muSF_IdCutTight' : [ jsonpath+'MuonID_Z_RunCD_Reco76X_Feb15.json' , 'MC_NUM_TightIDandIPCut_DEN_genTracks_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'] ,
    'muSF_IdMVALoose' : ['','',''], 
    'muSF_IdMVATight' : ['','',''], 
    'eleEff_HLT_RunC' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
    'eleEff_HLT_RunD4p2' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
    'eleEff_HLT_RunD4p3' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
    'eleSF_HLT_RunC' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
    'eleSF_HLT_RunD4p2' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
    'eleSF_HLT_RunD4p3' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
    'eleSF_IdCutLoose' : [jsonpath+'CutBasedID_LooseWP.json', 'CutBasedID_LooseWP', 'abseta_pt_ratio'],
    'eleSF_IdCutTight' : [jsonpath+'CutBasedID_TightWP.json', 'CutBasedID_TightWP', 'abseta_pt_ratio'],
    'eleSF_IdMVALoose' : [jsonpath+'ScaleFactor_egammaEff_WP80.json', 'ScaleFactor_egammaEff_WP80', 'eta_pt_ratio'],
    'eleSF_IdMVATight' : [jsonpath+'ScaleFactor_egammaEff_WP90.json', 'ScaleFactor_egammaEff_WP90', 'eta_pt_ratio'],
    'eleSF_IsoLoose' : ['','',''],
    'eleSF_IsoTight' : ['','',''],
    'eleSF_trk_eta' : ['','',''],
    #NEW
    'muSF_HLT_RunD4p2' : [ jsonpath+'SingleMuonTrigger_Z_RunBCD_prompt80X_7p65.json' , 'IsoMu22_OR_IsoTkMu22_PtEtaBins_Run273158_to_274093', 'abseta_pt_DATA' ],
    'muSF_HLT_RunD4p3' : [ jsonpath+'SingleMuonTrigger_Z_RunBCD_prompt80X_7p65.json' , 'IsoMu22_OR_IsoTkMu22_PtEtaBins_Run274094_to_276097', 'abseta_pt_DATA' ],
    'muSF_IsoLoose' : [ jsonpath+'MuonIso_Z_RunBCD_prompt80X_7p65.json' , 'MC_NUM_LooseRelIso_DEN_TightID_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
    'muSF_IsoTight' : [ jsonpath+'MuonIso_Z_RunBCD_prompt80X_7p65.json' , 'MC_NUM_TightRelIso_DEN_TightID_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
    'muSF_IdCutLoose' : [ jsonpath+'MuonID_Z_RunBCD_prompt80X_7p65.json' , 'MC_NUM_LooseID_DEN_genTracks_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
    'muSF_IdCutTight' : [ jsonpath+'MuonID_Z_RunBCD_prompt80X_7p65.json' , 'MC_NUM_TightIDandIPCut_DEN_genTracks_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
    'muSF_trk_eta' : [ jsonpath+'MuonTrkHIP_80X_Jul28.json' , 'ratio_eta', 'ratio_eta' ],
    }

correctors = {}
for name, conf in jsons.iteritems(): 
    correctors[name] = LeptonSF(conf[0], conf[1], conf[2])

for cut in ["IsoLoose", "IsoTight", "IdCutLoose", "IdCutTight", "IdMVALoose", "IdMVATight", "HLT_RunD4p3","HLT_RunD4p2","HLT_RunC"]:     
    leptonTypeVHbb.variables += [NTupleVariable("SF_"+cut, 
                                                lambda x, muCorr=correctors["muSF_"+cut], eleCorr=correctors["eleSF_"+cut] : muCorr.get_2D(x.pt(), x.eta())[0] if abs(x.pdgId()) == 13 else eleCorr.get_2D(x.pt(), x.eta())[0], 
                                                float, mcOnly=True, help="SF for lepton "+cut
                                                )]
    leptonTypeVHbb.variables += [NTupleVariable("SFerr_"+cut, 
                                                lambda x, muCorr=correctors["muSF_"+cut], eleCorr=correctors["eleSF_"+cut] : muCorr.get_2D(x.pt(), x.eta())[1] if abs(x.pdgId()) == 13 else eleCorr.get_2D(x.pt(), x.eta())[1], 
                                                float, mcOnly=True, help="SF error for lepton "+cut
                                                )]
for cut in ["trk_eta"]:     
    leptonTypeVHbb.variables += [NTupleVariable("SF_"+cut, 
                                                lambda x, muCorr=correctors["muSF_"+cut], eleCorr=correctors["eleSF_"+cut] : muCorr.get_1D(x.eta())[0] if abs(x.pdgId()) == 13 else eleCorr.get_1D(x.eta())[0], 
                                                float, mcOnly=True, help="SF for lepton "+cut
                                                )]
    leptonTypeVHbb.variables += [NTupleVariable("SFerr_"+cut, 
                                                lambda x, muCorr=correctors["muSF_"+cut], eleCorr=correctors["eleSF_"+cut] : muCorr.get_1D(x.eta())[1] if abs(x.pdgId()) == 13 else eleCorr.get_1D(x.eta())[1], 
                                                float, mcOnly=True, help="SF error for lepton "+cut
                                                )]
for cut in ["HLT_RunD4p3","HLT_RunD4p2","HLT_RunC"]:     
    leptonTypeVHbb.variables += [NTupleVariable("Eff_"+cut, 
                                                lambda x, muCorr=correctors["muEff_"+cut], eleCorr=correctors["eleEff_"+cut] : muCorr.get_2D(x.pt(), x.eta())[0] if abs(x.pdgId()) == 13 else eleCorr.get_2D(x.pt(), x.eta())[0], 
                                                float, mcOnly=True, help="SF for lepton "+cut
                                                )]
    leptonTypeVHbb.variables += [NTupleVariable("Efferr_"+cut, 
                                                lambda x, muCorr=correctors["muEff_"+cut], eleCorr=correctors["eleEff_"+cut] : muCorr.get_2D(x.pt(), x.eta())[1] if abs(x.pdgId()) == 13 else eleCorr.get_2D(x.pt(), x.eta())[1], 
                                                float, mcOnly=True, help="SF error for lepton "+cut
                                                )]


##------------------------------------------  
## FAT JET + Tau
##------------------------------------------  

# Four Vector + Nsubjettiness

fatjetTauType = NTupleObjectType("fatjettau",  baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("tau1",  lambda x : x.tau1, help="Nsubjettiness (1 axis)"),
    NTupleVariable("tau2",  lambda x : x.tau2, help="Nsubjettiness (2 axes)"),
    NTupleVariable("tau3",  lambda x : x.tau3, help="Nsubjettiness (3 axes)"),
])

 
##------------------------------------------  
## FAT JET
##------------------------------------------  

# Four Vector + Nsubjettiness + Hbb-Tag

fatjetType = NTupleObjectType("fatjet",  baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("tau1",  lambda x : x.tau1, help="Nsubjettiness (1 axis)"),
    NTupleVariable("tau2",  lambda x : x.tau2, help="Nsubjettiness (2 axes)"),
    NTupleVariable("tau3",  lambda x : x.tau3, help="Nsubjettiness (3 axes)"),
    
    # bb-tag output variable
    NTupleVariable("bbtag",  lambda x : x.bbtag, help="Hbb b-tag score"),

    ])


##------------------------------------------  
## Extended FAT JET
##------------------------------------------  

# Four Vector + Nsubjettiness + masses + Hbb-Tag

ak8FatjetType = NTupleObjectType("ak8fatjet",  baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("tau1",  lambda x : x.userFloat("NjettinessAK8:tau1"), help="Nsubjettiness (1 axis)"),
    NTupleVariable("tau2",  lambda x : x.userFloat("NjettinessAK8:tau2"), help="Nsubjettiness (2 axes)"),
    NTupleVariable("tau3",  lambda x : x.userFloat("NjettinessAK8:tau3"), help="Nsubjettiness (3 axes)"),

    NTupleVariable("msoftdrop",  lambda x : x.userFloat("ak8PFJetsCHSSoftDropMass"),  help="Softdrop Mass"),
    NTupleVariable("mpruned",    lambda x : x.userFloat("ak8PFJetsCHSPrunedMass"),    help="Pruned Mass"),
    NTupleVariable("mprunedcorr",    lambda x : x.mprunedcorr,    help="Pruned Mass L2+L3 corrected"),
    NTupleVariable("JEC_L2L3",    lambda x : x.JEC_L2L3,    help="L2+L3 correction factor for pruned mass"),	
    NTupleVariable("JEC_L1L2L3",    lambda x : x.JEC_L1L2L3,    help="L1+L2+L3 correction factor for ungroomed pt"),	
    NTupleVariable("JEC_L2L3Unc",    lambda x : x.JEC_L2L3Unc,    help="Unc L2+L3 correction factor for pruned mass"),
    NTupleVariable("JEC_L1L2L3Unc",    lambda x : x.JEC_L1L2L3Unc,    help="Unc L1+L2+L3 correction factor for ungroomed pt"),

    NTupleVariable("bbtag",  lambda x : x.bbtag, help="Hbb b-tag score"),
    NTupleVariable("id_Tight",  lambda x : (x.numberOfDaughters()>1 and x.neutralEmEnergyFraction() <0.90 and x.neutralHadronEnergyFraction()<0.90 and x.muonEnergyFraction()  < 0.8) and (x.eta>2.4 or (x.chargedEmEnergyFraction()<0.90 and x.chargedHadronEnergyFraction()>0 and x.chargedMultiplicity()>0)) , help="POG Tight jet ID lep veto"),
  # ID variables
    NTupleVariable("numberOfDaughters",  lambda x : x.numberOfDaughters(), help = "numberOfDaughters" ),
    NTupleVariable("neutralEmEnergyFraction",  lambda x : x.neutralEmEnergyFraction(), help = "neutralEmEnergyFraction" ),
    NTupleVariable("neutralHadronEnergyFraction",  lambda x : x.neutralHadronEnergyFraction(), help = "neutralHadronEnergyFraction" ),
    NTupleVariable("muonEnergyFraction",  lambda x : x.muonEnergyFraction(), help = "muonEnergyFraction" ),
    NTupleVariable("chargedEmEnergyFraction",  lambda x : x.chargedEmEnergyFraction(), help = "chargedEmEnergyFraction" ),
    NTupleVariable("chargedHadronEnergyFraction",  lambda x : x.chargedHadronEnergyFraction(), help = "chargedHadronEnergyFraction" ),
    NTupleVariable("chargedMultiplicity",  lambda x : x.chargedMultiplicity(), help = "chargedMultiplicity" ),


    NTupleVariable("Flavour", lambda x : x.partonFlavour(), int,     mcOnly=True, help="parton flavor as ghost matching"),
    NTupleVariable("BhadronFlavour", lambda x : x.jetFlavourInfo().getbHadrons().size(), int,     mcOnly=True, help="hadron flavour (ghost matching to B hadrons)"),
    NTupleVariable("ChadronFlavour", lambda x : x.jetFlavourInfo().getcHadrons().size(), int,     mcOnly=True, help="hadron flavour (ghost matching to C hadrons)"),	

    NTupleVariable("GenPt", lambda x : x.genJetFwdRef().pt() if (x.genJetFwdRef().isNonnull() and x.genJetFwdRef().isAvailable())  else -1., float, mcOnly=True, help="gen jet pt for JER computation"),
    
    # bb-tag input variables
    NTupleVariable("PFLepton_ptrel",   lambda x : x.PFLepton_ptrel, help="pt-rel of e/mu (for bb-tag)"),    
    NTupleVariable("z_ratio",          lambda x : x.z_ratio, help="z-ratio (for bb-tag)"),    
    NTupleVariable("PFLepton_IP2D",    lambda x : x.PFLepton_IP2D, help="lepton IP2D (for bb-tag)"),    
    NTupleVariable("nSL", lambda x : x.nSL, help="number of soft leptons (for bb-tag)"),    
    #NTupleVariable("trackSipdSig_3", lambda x : x.trackSip3dSig_3 , help=" bb-tag input as in 76x"),
    #NTupleVariable("trackSipdSig_2", lambda x : x.trackSip3dSig_2 , help=" bb-tag input as in 76x"),
    #NTupleVariable("trackSipdSig_1", lambda x : x.trackSip3dSig_1, help=" bb-tag input as in 76x"),
    #NTupleVariable("trackSipdSig_0", lambda x : x.trackSip3dSig_0, help=" bb-tag input as in 76x"),
    #NTupleVariable("trackSipdSig_1_0", lambda x : x.tau2_trackSip3dSig_0, help=" bb-tag input as in 76x"),
    #NTupleVariable("trackSipdSig_0_0", lambda x : x.tau1_trackSip3dSig_0, help=" bb-tag input as in 76x"),
    #NTupleVariable("trackSipdSig_1_1", lambda x : x.tau2_trackSip3dSig_1, help=" bb-tag input as in 76x"),
    #NTupleVariable("trackSipdSig_0_1", lambda x : x.tau1_trackSip3dSig_1, help=" bb-tag input as in 76x"),
    #NTupleVariable("trackSip2dSigAboveCharm_0", lambda x : x.trackSip2dSigAboveCharm_0, help=" bb-tag input as in 76x"),
    #NTupleVariable("trackSip2dSigAboveBottom_0", lambda x : x.trackSip2dSigAboveBottom_0, help=" bb-tag input as in 76x"),
    #NTupleVariable("trackSip2dSigAboveBottom_1", lambda x : x.trackSip2dSigAboveBottom_1, help=" bb-tag input as in 76x"),
    NTupleVariable("tau1_trackEtaRel_0", lambda x : x.tau2_trackEtaRel_0, help=" bb-tag input as in 76x"),
    NTupleVariable("tau1_trackEtaRel_1", lambda x : x.tau2_trackEtaRel_1, help=" bb-tag input as in 76x"),
    NTupleVariable("tau1_trackEtaRel_2", lambda x : x.tau2_trackEtaRel_2, help=" bb-tag input as in 76x"),
    NTupleVariable("tau0_trackEtaRel_0", lambda x : x.tau1_trackEtaRel_0, help=" bb-tag input as in 76x"),
    NTupleVariable("tau0_trackEtaRel_1", lambda x : x.tau1_trackEtaRel_1, help=" bb-tag input as in 76x"),
    NTupleVariable("tau0_trackEtaRel_2", lambda x : x.tau1_trackEtaRel_2, help=" bb-tag input as in 76x"),
    NTupleVariable("tau_vertexMass_0", lambda x : x.tau1_vertexMass, help=" bb-tag input as in 76x"),
    NTupleVariable("tau_vertexEnergyRatio_0", lambda x : x.tau1_vertexEnergyRatio, help=" bb-tag input as in 76x"),
    NTupleVariable("tau_vertexDeltaR_0", lambda x : x.tau1_vertexDeltaR, help=" bb-tag input as in 76x"),
    NTupleVariable("tau_flightDistance2dSig_0", lambda x : x.tau1_flightDistance2dSig, help=" bb-tag input as in 76x"),
    NTupleVariable("tau_vertexMass_1", lambda x : x.tau2_vertexMass, help=" bb-tag input as in 76x"),
    NTupleVariable("tau_vertexEnergyRatio_1", lambda x : x.tau2_vertexEnergyRatio, help=" bb-tag input as in 76x"),
    NTupleVariable("tau_flightDistance2dSig_1", lambda x : x.tau2_flightDistance2dSig, help=" bb-tag input as in 76x"),
    #NTupleVariable("jetNTracks", lambda x : x.jetNTracks, help=" bb-tag input as in 76x"),
    NTupleVariable("nSV", lambda x : x.nSV, help=" bb-tag input as in 76x"),


    

    ])


##------------------------------------------  
## Subjet
##------------------------------------------  

# Four Vector + b-Tag + JetID 

subjetType = NTupleObjectType("subjet",  baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("btag",    lambda x : x.btag, help="CVS IVF V2 btag-score"),
    NTupleVariable("jetID",    lambda x : x.jetID, help="Jet ID (loose) + pT/eta cuts"),
    NTupleVariable("fromFJ",  lambda x : x.fromFJ, help="assigns subjet to fatjet. index of fatjet. Use the matching fj collection - eg: ca15prunedsubjets and ca15pruned"),
],)

##------------------------------------------  
## PAT Subjet
##------------------------------------------  

# Four Vector + b-Tag from PAT

patSubjetType = NTupleObjectType("patsubjet",  baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("btag",  lambda x : x.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags"), help="CVS IVF V2 btag-score")])


##------------------------------------------  
## HEPTopTagger Candidate
##------------------------------------------  

# Four Vector + fW + Rmin + RminExp + Subjets

# The W/non-W assignment is done using the mass ratio in the HTT

httType = NTupleObjectType("htt",  baseObjectTypes = [ fourVectorType ], variables = [

    NTupleVariable("ptcal",   lambda x : x.ptcal,   help="pT (calibrated)"),
    NTupleVariable("etacal",  lambda x : x.etacal,  help="eta (calibrated)"),
    NTupleVariable("phical",  lambda x : x.phical,  help="phi (calibrated)"),
    NTupleVariable("masscal", lambda x : x.masscal, help="mass (calibrated)"),

    NTupleVariable("fRec",  lambda x : x.fRec, help="relative W width"),
    NTupleVariable("Ropt",  lambda x : x.Ropt, help="optimal value of R"),
    NTupleVariable("RoptCalc",  lambda x : x.RoptCalc, help="expected value of optimal R"),
    NTupleVariable("ptForRoptCalc",  lambda x : x.ptForRoptCalc, help="pT used for calculation of RoptCalc"),
    NTupleVariable("subjetIDPassed",  lambda x : x.subjetIDPassed, help="Do all the subjets pass jet id criteria?"),
    
    # Leading W Subjet (pt)
    NTupleVariable("sjW1ptcal",lambda x : x.sjW1ptcal,help = "Leading W Subjet pT (calibrated)"),
    NTupleVariable("sjW1pt",   lambda x : x.sjW1pt,   help = "Leading W Subjet pT"),
    NTupleVariable("sjW1eta",  lambda x : x.sjW1eta,  help = "Leading W Subjet eta"),
    NTupleVariable("sjW1phi",  lambda x : x.sjW1phi,  help = "Leading W Subjet phi"),
    NTupleVariable("sjW1masscal", lambda x : x.sjW1masscal, help = "Leading W Subjet mass (calibrated)"),
    NTupleVariable("sjW1mass", lambda x : x.sjW1mass, help = "Leading W Subjet mass"),
    NTupleVariable("sjW1btag", lambda x : x.sjW1btag, help = "Leading W Subjet btag"),
    # Second W Subjet (pt)
    NTupleVariable("sjW2ptcal", lambda x : x.sjW2ptcal,help = "Second Subjet pT (calibrated)"),
    NTupleVariable("sjW2pt",   lambda x : x.sjW2pt,   help = "Second Subjet pT"),
    NTupleVariable("sjW2eta",  lambda x : x.sjW2eta,  help = "Second Subjet eta"),
    NTupleVariable("sjW2phi",  lambda x : x.sjW2phi,  help = "Second Subjet phi"),
    NTupleVariable("sjW2masscal", lambda x : x.sjW2masscal, help = "Second Subjet mass (calibrated)"),
    NTupleVariable("sjW2mass", lambda x : x.sjW2mass, help = "Second Subjet mass"),
    NTupleVariable("sjW2btag", lambda x : x.sjW2btag, help = "Second Subjet btag"),
    # Non-W Subjet
    NTupleVariable("sjNonWptcal",lambda x : x.sjNonWptcal,help = "Non-W Subjet pT (calibrated)"),
    NTupleVariable("sjNonWpt",   lambda x : x.sjNonWpt,   help = "Non-W Subjet pT"),
    NTupleVariable("sjNonWeta",  lambda x : x.sjNonWeta,  help = "Non-W Subjet eta"),
    NTupleVariable("sjNonWphi",  lambda x : x.sjNonWphi,  help = "Non-W Subjet phi"),
    NTupleVariable("sjNonWmasscal", lambda x : x.sjNonWmasscal, help = "Non-W Subjet mass (calibrated)"),
    NTupleVariable("sjNonWmass", lambda x : x.sjNonWmass, help = "Non-W Subjet mass"),
    NTupleVariable("sjNonWbtag", lambda x : x.sjNonWbtag, help = "Non-W Subjet btag"),
    ])
   

##------------------------------------------  
## SECONDARY VERTEX CANDIDATE
##------------------------------------------  
  
svType = NTupleObjectType("sv", baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("charge",   lambda x : x.charge(), int),
    NTupleVariable("ntracks", lambda x : x.numberOfDaughters(), int, help="Number of tracks (with weight > 0.5)"),
    NTupleVariable("chi2", lambda x : x.vertexChi2(), help="Chi2 of the vertex fit"),
    NTupleVariable("ndof", lambda x : x.vertexNdof(), help="Degrees of freedom of the fit, ndof = (2*ntracks - 3)" ),
    NTupleVariable("dxy",  lambda x : x.dxy.value(), help="Transverse distance from the PV [cm]"),
    NTupleVariable("edxy", lambda x : x.dxy.error(), help="Uncertainty on the transverse distance from the PV [cm]"),
    NTupleVariable("ip3d",  lambda x : x.d3d.value(), help="3D distance from the PV [cm]"),
    NTupleVariable("eip3d", lambda x : x.d3d.error(), help="Uncertainty on the 3D distance from the PV [cm]"),
    NTupleVariable("sip3d", lambda x : x.d3d.significance(), help="S_{ip3d} with respect to PV (absolute value)"),
    NTupleVariable("cosTheta", lambda x : x.cosTheta, help="Cosine of the angle between the 3D displacement and the momentum"),
    NTupleVariable("jetPt",  lambda x : x.jet.pt() if x.jet != None else 0, help="pT of associated jet"),
    NTupleVariable("jetBTag",  lambda x : x.jet.btag('pfCombinedInclusiveSecondaryVertexV2BJetTags') if x.jet != None else -99, help="CSV b-tag of associated jet"),
    NTupleVariable("mcMatchNTracks", lambda x : x.mcMatchNTracks, int, mcOnly=True, help="Number of mc-matched tracks in SV"),
    NTupleVariable("mcMatchNTracksHF", lambda x : x.mcMatchNTracksHF, int, mcOnly=True, help="Number of mc-matched tracks from b/c in SV"),
    NTupleVariable("mcMatchFraction", lambda x : x.mcMatchFraction, mcOnly=True, help="Fraction of mc-matched tracks from b/c matched to a single hadron (or -1 if mcMatchNTracksHF < 2)"),
    NTupleVariable("mcFlavFirst", lambda x : x.mcFlavFirst, int, mcOnly=True, help="Flavour of last ancestor with maximum number of matched daughters"),
    NTupleVariable("mcFlavHeaviest", lambda x : x.mcFlavHeaviest, int, mcOnly=True, help="Flavour of heaviest hadron with maximum number of matched daughters"),
])


##------------------------------------------  
## Trigger object type
##------------------------------------------  

triggerObjectsType = NTupleObjectType("triggerObjects",  baseObjectTypes = [ fourVectorType ], variables = [
])
triggerObjectsOnlyPtType = NTupleObjectType("triggerObjects",  baseObjectTypes = [ ], variables = [
    NTupleVariable("pt", lambda x : x.pt(), float, mcOnly=False, help="trigger object pt"),
])
triggerObjectsNothingType = NTupleObjectType("triggerObjects",  baseObjectTypes = [ ], variables = [
])

##------------------------------------------  
## Heavy flavour hadron
##------------------------------------------  


heavyFlavourHadronType = NTupleObjectType("heavyFlavourHadron", baseObjectTypes = [ genParticleType ], variables = [
    NTupleVariable("flav", lambda x : x.flav, int, mcOnly=True, help="Flavour"),
    NTupleVariable("sourceId", lambda x : x.sourceId, int, mcOnly=True, help="pdgId of heaviest mother particle (stopping at the first one heaviest than 175 GeV)"),
    NTupleVariable("svMass",   lambda x : x.sv.mass() if x.sv else 0, help="SV: mass"),
    NTupleVariable("svPt",   lambda x : x.sv.pt() if x.sv else 0, help="SV: pt"),
    NTupleVariable("svCharge",   lambda x : x.sv.charge() if x.sv else -99., int, help="SV: charge"),
    NTupleVariable("svNtracks", lambda x : x.sv.numberOfDaughters() if x.sv else 0, int, help="SV: Number of tracks (with weight > 0.5)"),
    NTupleVariable("svChi2", lambda x : x.sv.vertexChi2() if x.sv else -99., help="SV: Chi2 of the vertex fit"),
    NTupleVariable("svNdof", lambda x : x.sv.vertexNdof() if x.sv else -99., help="SV: Degrees of freedom of the fit, ndof = (2*ntracks - 3)" ),
    NTupleVariable("svDxy",  lambda x : x.sv.dxy.value() if x.sv else -99., help="SV: Transverse distance from the PV [cm]"),
    NTupleVariable("svEdxy", lambda x : x.sv.dxy.error() if x.sv else -99., help="SV: Uncertainty on the transverse distance from the PV [cm]"),
    NTupleVariable("svIp3d",  lambda x : x.sv.d3d.value() if x.sv else -99., help="SV: 3D distance from the PV [cm]"),
    NTupleVariable("svEip3d", lambda x : x.sv.d3d.error() if x.sv else -99., help="SV: Uncertainty on the 3D distance from the PV [cm]"),
    NTupleVariable("svSip3d", lambda x : x.sv.d3d.significance() if x.sv else -99., help="SV: S_{ip3d} with respect to PV (absolute value)"),
    NTupleVariable("svCosTheta", lambda x : x.sv.cosTheta if x.sv else -99., help="SV: Cosine of the angle between the 3D displacement and the momentum"),
    NTupleVariable("jetPt",  lambda x : x.jet.pt() if x.jet != None else 0, help="Jet: pT"),
    NTupleVariable("jetBTag",  lambda x : x.jet.btag('pfCombinedInclusiveSecondaryVertexV2BJetTags') if x.jet != None else -99, help="CSV b-tag of associated jet"),
])
shiftedMetType= NTupleObjectType("shiftedMetType", baseObjectTypes=[twoVectorType], variables=[
    NTupleVariable("sumEt", lambda x : x.sumEt() ),
])

primaryVertexType = NTupleObjectType("primaryVertex", variables = [
    NTupleVariable("x",    lambda x : x.x()),
    NTupleVariable("y",   lambda x : x.y()),
    NTupleVariable("z",   lambda x : x.z()),
    NTupleVariable("isFake",   lambda x : x.isFake()),
    NTupleVariable("ndof",   lambda x : x.ndof()),
    NTupleVariable("Rho",   lambda x : x.position().Rho()),
    NTupleVariable("score",  lambda x : x.score),
])

genTauJetType = NTupleObjectType("genTauJet", baseObjectTypes = [ genParticleType ], variables = [
    NTupleVariable("decayMode", lambda x : x.decayMode, int, mcOnly=True, help="Generator level tau decay mode"),
])

genTopType = NTupleObjectType("genTopType", baseObjectTypes = [ genParticleType ], variables = [
    NTupleVariable("decayMode", lambda x : x.decayMode, int, mcOnly=True, help="Generator level top decay mode: 0=leptonic, 1=hadronic, -1=not known"),
])

genJetType = NTupleObjectType("genJet", baseObjectTypes = [ genParticleType ], variables = [
    NTupleVariable("numBHadrons", lambda x : getattr(x,"numBHadronsBeforeTop",-1), int, mcOnly=True, help="number of matched b hadrons before top quark decay"),
    NTupleVariable("numCHadrons", lambda x : getattr(x,"numCHadronsBeforeTop",-1), int, mcOnly=True, help="number of matched c hadrons before top quark decay"),
    NTupleVariable("numBHadronsFromTop", lambda x : getattr(x,"numBHadronsFromTop",-1), int, mcOnly=True, help="number of matched b hadrons from top quark decay"),
    NTupleVariable("numCHadronsFromTop", lambda x : getattr(x,"numCHadronsFromTop",-1), int, mcOnly=True, help="number of matched c hadrons from top quark decay"),
    NTupleVariable("numBHadronsAfterTop", lambda x : getattr(x,"numBHadronsAfterTop",-1), int, mcOnly=True, help="number of matched b hadrons after top quark decay"),
    NTupleVariable("numCHadronsAfterTop", lambda x : getattr(x,"numCHadronsAfterTop",-1), int, mcOnly=True, help="number of matched c hadrons after top quark decay"),
    NTupleVariable("wNuPt", lambda x : (x.p4()+x.nu).pt() if hasattr(x,"nu") else x.p4().pt() ,float, mcOnly=True, help="pt of jet adding back the neutrinos"),
    NTupleVariable("wNuEta", lambda x : (x.p4()+x.nu).eta() if hasattr(x,"nu") else x.p4().eta() ,float, mcOnly=True, help="eta of jet adding back the neutrinos"),
    NTupleVariable("wNuPhi", lambda x : (x.p4()+x.nu).phi() if hasattr(x,"nu") else x.p4().phi() ,float, mcOnly=True, help="phi of jet adding back the neutrinos"),
    NTupleVariable("wNuM", lambda x : (x.p4()+x.nu).M() if hasattr(x,"nu") else x.p4().M() ,float, mcOnly=True, help="mass of jet adding back the neutrinos"),

])

softActivityType = NTupleObjectType("softActivity", baseObjectTypes = [  ], variables = [
                 NTupleVariable("njets2", lambda sajets: len([ x for x in sajets if x.pt()> 2 ] ), int, help="number of jets from soft activity with pt>2Gev"),
                 NTupleVariable("njets5", lambda sajets: len([ x for x in sajets if x.pt()> 5 ] ), int, help="number of jets from soft activity with pt>5Gev"),
                 NTupleVariable("njets10", lambda sajets: len([ x for x in sajets if x.pt()> 10 ] ), int, help="number of jets from soft activity with pt>10Gev"),
                 NTupleVariable("HT", lambda sajets: sum([x.pt() for x in sajets],0.0), float, help="sum pt of sa jets"),
])

def ptRel(p4,axis):
    a=ROOT.TVector3(axis.Vect().X(),axis.Vect().Y(),axis.Vect().Z())
    o=ROOT.TLorentzVector(p4.Px(),p4.Py(),p4.Pz(),p4.E())
    return o.Perp(a)

