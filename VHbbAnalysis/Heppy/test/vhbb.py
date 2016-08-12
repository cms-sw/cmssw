#! /usr/bin/env python
fastObjects=True

#Switch to True to produce x1, x2, id1, id2, pdf scale
doPDFVars = False

import ROOT
from DataFormats.FWLite import *
import PhysicsTools.HeppyCore.framework.config as cfg
from VHbbAnalysis.Heppy.vhbbobj import *
from PhysicsTools.HeppyCore.utils.deltar import deltaPhi
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer  import * 


import logging
logging.basicConfig(level=logging.ERROR)

import os

cfg.Analyzer.nosubdir = True

treeProducer= cfg.Analyzer(
	class_object=AutoFillTreeProducer, 
	defaultFloatType = "F",
	verbose=False, 
	vectorTree = True,
        globalVariables	= [
                 NTupleVariable("puWeightUp", lambda ev : getattr(ev,"puWeightPlus",1.), help="Pileup up variation",mcOnly=True),
                 NTupleVariable("puWeightDown", lambda ev : getattr(ev,"puWeightMinus",1.), help="Pileup down variation",mcOnly=True),
                 NTupleVariable("json", lambda ev : getattr(ev,"json",True), help="Passing json selection"),
                 NTupleVariable("json_silver", lambda ev : getattr(ev,"json_silver",True), help="Passing silver json selection"),
                 NTupleVariable("nPU0", lambda ev : [bx.nPU() for bx in  ev.pileUpInfo if bx.getBunchCrossing()==0][0], help="nPU in BX=0",mcOnly=True),
                 NTupleVariable("nPVs", lambda ev : len(ev.goodVertices), help="total number of good PVs"),
		 NTupleVariable("Vtype", lambda ev : ev.Vtype, help="Event classification"),
		 NTupleVariable("VtypeSim", lambda ev : ev.VtypeSim, help="Event classification",mcOnly=True),
		 NTupleVariable("VMt", lambda ev : ev.V.goodMt, help="Transverse mass of the vector boson"),
		 NTupleVariable("HVdPhi", lambda ev : deltaPhi(ev.V.phi(),ev.H.phi()), help="Delta phi between Higgs and Z/W"),
		 NTupleVariable("fakeMET_sumet", lambda ev : ev.fakeMET.sumet, help="Fake SumET from Zmumu events removing muons"),
    	         NTupleVariable("bx",  lambda ev: ev.input.eventAuxiliary().bunchCrossing(), int, help="bunch crossing number"),
		 NTupleVariable("caloMetPt",  lambda ev: ev.met.caloMETPt(), float, help="calo met pt "),
		 NTupleVariable("caloMetPhi",  lambda ev: ev.met.caloMETPhi(), float, help="calo met phi"),
		 NTupleVariable("rho",  lambda ev: ev.rho, float, help="kt6PFJets rho"),
		 NTupleVariable("rhoN",  lambda ev: ev.rhoN, float, help="rho with neutrals only"),
		 NTupleVariable("rhoCHPU",  lambda ev: ev.rhoCHPU, float, help="rho with charged pileup particles"),
		 NTupleVariable("rhoCentral",  lambda ev: ev.rho, float, help="rho central"),
          
		 NTupleVariable("deltaR_jj",  lambda ev: deltaR(ev.hJets[0].eta(),ev.hJets[0].phi(),ev.hJets[1].eta(),ev.hJets[1].phi()) if len(ev.hJets) > 1 else -1, float, help="deltaR higgsJets"),
		 NTupleVariable("lheNj",  lambda ev: ev.lheNj, float,mcOnly=True, help="number of jets at LHE level"),
                 NTupleVariable("lheNb",  lambda ev: ev.lheNb, float,mcOnly=True, help="number of b-jets at LHE level"),
                 NTupleVariable("lheNc",  lambda ev: ev.lheNc, float,mcOnly=True, help="number of c-jets at LHE level"),
                 NTupleVariable("lheNg",  lambda ev: ev.lheNg, float,mcOnly=True, help="number of gluon jets at LHE level"),
                 NTupleVariable("lheNl",  lambda ev: ev.lheNl, float,mcOnly=True, help="number of light(uds) jets at LHE level"),
		 NTupleVariable("lheV_pt",  lambda ev: ev.lheV_pt, float,mcOnly=True, help="Vector pT at LHE level"),
                 NTupleVariable("lheHT",  lambda ev: ev.lheHT, float,mcOnly=True, help="HT at LHE level"),
                 #NTupleVariable("LHE_originalWeight",  lambda ev: ev.LHE_originalWeight, float, mcOnly=True, help="LHE original weight (for normalisation of LHE_weights)"),
                 NTupleVariable("genTTHtoTauTauDecayMode", lambda ev: ev.genTTHtoTauTauDecayMode, int,mcOnly=True, help="gen level ttH, H -> tautau decay mode"),        		 
		#Soft Activity vars
#                 NTupleVariable("totSoftActivityJets2", lambda ev: len([ x for x in ev.softActivityJets if x.pt()> 2 ] ), int, help="number of jets from soft activity with pt>2Gev"),
 #                NTupleVariable("totSoftActivityJets5", lambda ev: len([ x for x in ev.softActivityJets if x.pt()> 5 ] ), int, help="number of jets from soft activity with pt>5Gev"),
  #               NTupleVariable("totSoftActivityJets10", lambda ev: len([ x for x in ev.softActivityJets if x.pt()> 10 ] ), int, help="number of jets from soft activity with pt>10Gev"),
                 NTupleVariable("ttCls",  lambda ev: getattr(ev, "ttbarCls", -1), float,mcOnly=True, help="ttbar classification via GenHFHadronMatcher"),
                 NTupleVariable("heavyFlavourCategory",  lambda ev: getattr(ev, "ttbarCategory", -1), float,mcOnly=True, help="ttbar classification via GenHFHadronMatcher and the official GenTtbarClassifier code"),
		 NTupleVariable("mhtJet30",  lambda ev : ev.mhtJet30, help="mht with jets30"),
		 NTupleVariable("mhtPhiJet30",  lambda ev : ev.mhtPhiJet30, help="mht phi with jets30"),
		 NTupleVariable("htJet30",  lambda ev : ev.htJet30, help="ht  with jets30"),
                 NTupleVariable("met_sig",  lambda ev : ev.met.significance(), help="met significance from MET::significance() method"),
		 NTupleVariable("met_rawpt",  lambda ev : ev.met.uncorPt(), help="raw met"),
		 NTupleVariable("metPuppi_pt",  lambda ev : ev.metPuppi.pt(), help="met from Puppi"),
		 NTupleVariable("metPuppi_phi",  lambda ev : ev.metPuppi.phi(), help="met phi from Puppi"),
		 NTupleVariable("metPuppi_rawpt",  lambda ev : ev.metPuppi.uncorPt(), help="raw met from Puppi"),
		 NTupleVariable("metType1p2_pt",  lambda ev : ev.met.shiftedPt(12,2), help="type1type2met"),
#		 NTupleVariable("metNoPU_pt",  lambda ev : ev.metNoPU.pt(), help="PFnoPU E_{T}^{miss}"),
#		 NTupleVariable("metNoPU_phi",  lambda ev : ev.metNoPU.phi(), help="phi of PFnoPU E_{T}^{miss}"),
		 NTupleVariable("tkMet_pt",  lambda ev : ev.tkMet.pt(), help="E_{T}^{miss} from tracks"),
		 NTupleVariable("tkMet_phi",  lambda ev : ev.tkMet.phi(), help="phi of E_{T}^{miss} from tracks"),
		 NTupleVariable("tkMetPVchs_pt",  lambda ev : ev.tkMetPVchs.pt(), help="E_{T}^{miss} from tracks"),
		 NTupleVariable("tkMetPVchs_phi",  lambda ev : ev.tkMetPVchs.phi(), help="phi of E_{T}^{miss} from tracks"),
		 NTupleVariable("isrJetVH",  lambda ev : ev.isrJetVH, help="Index of ISR jet in VH"),
		 NTupleVariable("Flag_hbheIsoFilter",  lambda ev : ev.hbheFilterIso, help="hbheFilterIso, after rerun"),
		 NTupleVariable("Flag_hbheFilterNew",  lambda ev : ev.hbheFilterNew, help="hbheFilterIso, after rerun"),
		 NTupleVariable("simPrimaryVertex_z", lambda ev: ev.genvertex, float,mcOnly=True, help="z coordinate of the simulated primary vertex"),
		 NTupleVariable("genHiggsDecayMode", lambda ev: ev.genHiggsDecayMode, float, mcOnly=True, help="decay mode of the Higgs boson"),
		 NTupleVariable("triggerEmulationWeight", lambda ev: ev.triggerEmulationWeight, float, mcOnly=True, help="Emulates SL/DL triggers"),
	],
	globalObjects = {
          "met"    : NTupleObject("met",     metType, help="PF E_{T}^{miss}, after default type 1 corrections"),
          "fakeMET"    : NTupleObject("fakeMET", fourVectorType, help="fake MET in Zmumu event obtained removing the muons"),
          "H"    : NTupleObject("H", fourVectorType, help="higgs"),
          "H_reg"    : NTupleObject("H_reg", fourVectorType, help="regressed higgs"),
          "H_reg_corrJECUp"    : NTupleObject("H_reg_corrJECUp", fourVectorType, help="regressed higgs for JECUp"),
          "H_reg_corrJECDown"    : NTupleObject("H_reg_corrJECDown", fourVectorType, help="regressed higgs for JECDown"),
          "H_reg_corrJERUp"    : NTupleObject("H_reg_corrJERUp", fourVectorType, help="regressed higgs for JERUp"),
          "H_reg_corrJERDown"    : NTupleObject("H_reg_corrJERDown", fourVectorType, help="regressed higgs for JERDown"),
	  "HCSV"    : NTupleObject("HCSV", fourVectorType, help="higgs CSV selection"),
	  "HCSV_reg"    : NTupleObject("HCSV_reg", fourVectorType, help="regresses higgs CSV selection"),
          "HCSV_reg_corrJECUp"    : NTupleObject("HCSV_reg_corrJECUp", fourVectorType, help="reg_corrresses higgs for JECUp CSV selection"),
          "HCSV_reg_corrJECDown"    : NTupleObject("HCSV_reg_corrJECDown", fourVectorType, help="reg_corrresses higgs for JECDown CSV selection"),
          "HCSV_reg_corrJERUp"    : NTupleObject("HCSV_reg_corrJERUp", fourVectorType, help="reg_corrresses higgs for JERUp CSV selection"),
          "HCSV_reg_corrJERDown"    : NTupleObject("HCSV_reg_corrJERDown", fourVectorType, help="regressed higgs for JERDown CSV selection"),
	  "HCMVAV2"    : NTupleObject("HCMVAV2", fourVectorType, help="higgs CMVAV2 selection"),
	  "HCMVAV2_reg"    : NTupleObject("HCMVAV2_reg", fourVectorType, help="regresses higgs CMVAV2 selection"),
          "HCMVAV2_reg_corrJECUp"    : NTupleObject("HCMVAV2_reg_corrJECUp", fourVectorType, help="reg_corrresses higgs for JECUp CMVAV2 selection"),
          "HCMVAV2_reg_corrJECDown"    : NTupleObject("HCMVAV2_reg_corrJECDown", fourVectorType, help="reg_corrresses higgs for JECDown CMVAV2 selection"),
          "HCMVAV2_reg_corrJERUp"    : NTupleObject("HCMVAV2_reg_corrJERUp", fourVectorType, help="reg_corrresses higgs for JERUp CMVAV2 selection"),
          "HCMVAV2_reg_corrJERDown"    : NTupleObject("HCMVAV2_reg_corrJERDown", fourVectorType, help="regressed higgs for JERDown CMVAV2 selection"),
          "HaddJetsdR08"    : NTupleObject("HaddJetsdR08", fourVectorType, help="higgs with cen jets added if dR<0.8 from hJetsCSV selection"),
          "V"    : NTupleObject("V", fourVectorType, help="z or w"),
          "softActivityJets"    : NTupleObject("softActivity", softActivityType, help="VBF soft activity variables"),
          "softActivityVHJets"    : NTupleObject("softActivityVH", softActivityType, help="VH soft activity variables"),
        },
	collections = {
		#standard dumping of objects
   	        "selectedLeptons" : NTupleCollection("selLeptons", leptonTypeVHbb, 8, help="Leptons after the preselection"),
#   	        "inclusiveLeptons" : NTupleCollection("incLeptons", leptonTypeVHbb, 8, help="Leptons after the preselection"),
		#old style stuff, can be removed at some point
   	        "vLeptons" : NTupleCollection("vLeptons", leptonTypeVHbb, 8, help="Leptons after the preselection"),
   	        "aLeptons" : NTupleCollection("aLeptons", leptonTypeVHbb, 8, help="Additional leptons, not passing the preselection"),
# now store only indices, this lines are left commented for possible debugging
#	        "hJets"       : NTupleCollection("hJets",     jetTypeVHbb, 8, sortDescendingBy = lambda jet : jet.btag('combinedSecondaryVertexBJetTags'),help="Higgs jets"),
#	        "aJets"       : NTupleCollection("aJets",     jetTypeVHbb, 8, sortDescendingBy = lambda jet : jet.btag('combinedSecondaryVertexBJetTags'),help="Additional jets"),
                "hjidx"       : NTupleCollection("hJidx",    objectInt, 2,help="Higgs jet indices"),
                "hjidxDiJetPtByCSV"       : NTupleCollection("hJidx_sortcsv",    objectInt, 2,help="Higgs jet indices within hJets with CSV sorting "),
                "ajidx"       : NTupleCollection("aJidx",    objectInt, 8,help="additional jet indices"),
                "hjidxCSV"       : NTupleCollection("hJCidx",    objectInt, 2,help="Higgs jet indices CSV"),
                "ajidxCSV"       : NTupleCollection("aJCidx",    objectInt, 8,help="additional jet indices CSV"),
                "hjidxCMVAV2"       : NTupleCollection("hJCMVAV2idx",    objectInt, 2,help="Higgs jet indices CMVAV2"),
                "ajidxCMVAV2"       : NTupleCollection("aJCMVAV2idx",    objectInt, 8,help="additional jet indices CMVAV2"),
                "cleanJetsAll"       : NTupleCollection("Jet",     jetTypeVHbb, 25, help="Cental+fwd jets after full selection and cleaning, sorted by b-tag"),
                "hjidxaddJetsdR08"       : NTupleCollection("hjidxaddJetsdR08",    objectInt, 5,help="Higgs jet indices with Higgs formed adding cen jets if dR<0.8 from hJetsCSV"),
                "ajidxaddJetsdR08"       : NTupleCollection("ajidxaddJetsdR08",    objectInt, 8,help="additional jet indices with Higgs formed adding cen jets if dR<0.8 from hJetsCSV"),
		"dRaddJetsdR08"       : NTupleCollection("dRaddJetsdR08",    objectFloat, 5,help="dR of add jet with Higgs formed adding cen jets if dR<0.8 from hJetsCSV"),        
#                "discardedJets"       : NTupleCollection("DiscardedJet",     jetTypeVHbb, 15, help="jets that were discarded"),
                "inclusiveTaus"  : NTupleCollection("TauGood", tauTypeVHbb, 25, help="Taus after the preselection"),
                "softActivityJets"    : NTupleCollection("softActivityJets", fourVectorType, 5, help="jets made for soft activity"),
                "softActivityVHJets"    : NTupleCollection("softActivityVHJets", fourVectorType, 5, help="jets made for soft activity VH version"),
                "goodVertices"    : NTupleCollection("primaryVertices", primaryVertexType, 4, help="first four PVs"), 

		#dump of gen objects
                #"generatorSummary"    : NTupleCollection("GenSummary", genParticleWithLinksType, 30, help="Generator summary, see description in Heppy GeneratorAnalyzer",mcOnly=True),
                "genJets"    : NTupleCollection("GenJet",   genJetType, 15, help="Generated jets with hadron matching, sorted by pt descending",filter=lambda x: x.pt() > 20,mcOnly=True),
                "genHiggsSisters"    : NTupleCollection("GenHiggsSisters",     genParticleType, 4, help="Sisters of the Higgs bosons",mcOnly=True),
                "gentopquarks"    : NTupleCollection("GenTop",     genTopType, 4, help="Generated top quarks from hard scattering",mcOnly=True),
                "genallstatus2bhadrons"    : NTupleCollection("GenStatus2bHad",     genParticleType, 15, help="Generated Status 2 b Hadrons",mcOnly=True),
                "gennusFromTop"    : NTupleCollection("GenNuFromTop",     genParticleType, 4, help="Generated neutrino from t->W decay",mcOnly=True),
                "genbquarksFromH"      : NTupleCollection("GenBQuarkFromH",  genParticleType, 4, help="Generated bottom quarks from Higgs decays",mcOnly=True),
                "genbquarksFromTop"      : NTupleCollection("GenBQuarkFromTop",  genParticleType, 4, help="Generated bottom quarks from top decays",mcOnly=True),
                "genbquarksFromHafterISR"      : NTupleCollection("GenBQuarkFromHafterISR",  genParticleType, 4, help="Generated bottom quarks from Higgs decays",mcOnly=True),
                "genwzquarks"     : NTupleCollection("GenWZQuark",   genParticleType, 6, help="Generated quarks from W/Z decays",mcOnly=True),
                "genleps"         : NTupleCollection("GenLep",     genParticleType, 4, help="Generated leptons from W/Z decays",mcOnly=True),
                "gentaus"         : NTupleCollection("GenTaus",     genParticleType, 4, help="Generated taus",mcOnly=True),
		"genlepsRecovered"         : NTupleCollection("GenLepRecovered",     genParticleType, 4, help="Generated leptons from recovered W/Z decays",mcOnly=True),
                "gentausRecovered"         : NTupleCollection("GenTausRecovered",     genParticleType, 4, help="Generated taus from recovered W/Z decays",mcOnly=True),
                "genlepsFromTop"         : NTupleCollection("GenLepFromTop",     genParticleType, 4, help="Generated leptons from t->W decays",mcOnly=True),
                "gentauleps"      : NTupleCollection("GenLepFromTau", genParticleType, 6, help="Generated leptons from decays of taus from W/Z decays",mcOnly=True),
		"gentaulepsRecovered"      : NTupleCollection("GenLepFromTauRecovered", genParticleType, 6, help="Generated leptons from decays of taus from recovered W/Z decays",mcOnly=True),	
		"genHiggsBosons"   : NTupleCollection("GenHiggsBoson", genParticleType, 4, help="Generated Higgs boson ",mcOnly=True),
		#"genZbosonsToLL"  : NTupleCollection("GenZbosonsToLL", genParticleType, 6, help="Generated W or Z bosons decaying to LL"),
		#"genWbosonsToLL"  : NTupleCollection("GenWbosonsToLL", genParticleType, 6, help="Generated W or Z bosons decaying to LL"),
		"genvbosons"       : NTupleCollection("GenVbosons", genParticleType, 6, help="Generated W or Z bosons, mass > 30",mcOnly=True),
                "genvbosonsRecovered"  : NTupleCollection("GenVbosonsRecovered", genParticleType, 6, help="Generated W or Z bosons recovered from daughters, mass > 30",mcOnly=True),
		"pileUpVertex_z"       : NTupleCollection("pileUpVertex_z",    objectFloat, 5,help="z position of hardest pile-up collisions",mcOnly=True),        
		"pileUpVertex_ptHat"   : NTupleCollection("pileUpVertex_ptHat",    objectFloat, 5,help="z position of hardest pile-up collisions",mcOnly=True),        
		"LHE_weights_scale"       : NTupleCollection("LHE_weights_scale",   weightsInfoType , 6 ,help="LHE weights for scale variation", mcOnly=True),        
		"LHE_weights_pdf"       : NTupleCollection("LHE_weights_pdf",   weightsInfoType , 103 ,help="LHE weights for pdf variation (NNPDF)", mcOnly=True),        

	}
	)

#Create shifted MET Ntuples
metNames={y.__get__(ROOT.pat.MET):x for x,y in ROOT.pat.MET.__dict__.items() if  (x[-2:]=="Up" or x[-4:]=="Down")}
print "met Names", metNames
shifted_met_keys = ["met_shifted_{0}".format(n) for n in range(12)] #we do not need noShift I gueess
shifted_met_names = ["met_shifted_%s"%metNames[n] for n in range(12)] #we do not need noShift I gueess
shifted_mets = {mk: NTupleObject(nm, shiftedMetType, help="PF E_{T}^{miss}, after default type 1 corrections, shifted with %s" %mk) for mk,nm in zip(shifted_met_keys,shifted_met_names)}
treeProducer.globalObjects.update(shifted_mets)

btag_weights = {}
for algo in ["CSV", "CMVAV2"]:
	for syst in ["central", "up_jes", "down_jes", "up_lf", "down_lf", "up_hf", "down_hf", "up_hfstats1", "down_hfstats1", "up_hfstats2", "down_hfstats2", "up_lfstats1", "down_lfstats1", "up_lfstats2", "down_lfstats2", "up_cferr1", "down_cferr1", "up_cferr2", "down_cferr2"]:
		syst_name = "" if syst=="central" else ("_"+syst) 
		btag_weights["btagWeight"+algo+syst_name] = NTupleVariable("btagWeight"+algo+syst_name,
									   lambda ev, get_event_SF=get_event_SF, syst=syst, algo=algo, btag_calibrators=btag_calibrators : get_event_SF(ev.cleanJetsAll, syst, algo, btag_calibrators)
									   , float, mcOnly=True, help="b-tag "+algo+"continuous  weight, variating "+syst
									   )
treeProducer.globalVariables += list(btag_weights.values())

'''

for syst in ["JES", "LF", "HF", "HFStats1", "HFStats2", "LFStats1", "LFStats2", "cErr1", "cErr2"]:
	for sdir in ["Up", "Down"]:
		name = "bTagWeight"+syst+sdir
		btag_weights[name] = NTupleVariable("bTagWeight_" + syst + sdir,
			lambda ev, sname=syst+sdir: bweightcalc.calcEventWeight(
				ev.cleanJetsAll, kind="final", systematic=sname
			), float, mcOnly=True, help="b-tag CSV weight, variating "+syst+" "+sdir
		)
btag_weights["bTagWeight"] = NTupleVariable("bTagWeight",
	lambda ev: bweightcalc.calcEventWeight(
		ev.cleanJetsAll, kind="final", systematic="nominal"
	), float ,mcOnly=True, help="b-tag CSV weight, nominal"
)
#print list(btag_weights.values())
treeProducer.globalVariables += list(btag_weights.values())
'''

# Lepton Analyzer, take its default config and fix loose iso consistent with tight definition
from PhysicsTools.Heppy.analyzers.objects.LeptonAnalyzer import LeptonAnalyzer
LepAna = LeptonAnalyzer.defaultConfig
#LepAna.doElectronScaleCorrections = {
#	'data' : 'EgammaAnalysis/ElectronTools/data/76X_16DecRereco_2015', 
#	'GBRForest' : [os.environ['CMSSW_BASE']+'/src/VHbbAnalysis/Heppy/data/egamma/egamma_epComb_GBRForest_76X.root','gedelectron_p4combination_25ns'], 
#	'isSync' : False
#	}
LepAna.mu_isoCorr  = "deltaBeta"
#LepAna.loose_muon_isoCut = lambda muon : muon.relIso04 < 0.25 
LepAna.loose_muon_isoCut = lambda muon : muon.relIso04 < 0.4
LepAna.loose_muon_pt = 10.
LepAna.ele_isoCorr = "rhoArea"
LepAna.loose_electron_isoCut = lambda electron : electron.relIso03 < 0.4 
#LepAna.loose_electron_isoCut = lambda electron : ele_mvaEleID_Trig_preselection(electron)
LepAna.loose_electron_pt = 10.
#LepAna.loose_electron_id = "mvaEleID-Spring15-25ns-Trig-V1-wp90"
LepAna.doMiniIsolation = True

from PhysicsTools.Heppy.analyzers.objects.VertexAnalyzer import VertexAnalyzer
VertexAna = VertexAnalyzer.defaultConfig
VertexAna.keepFailingEvents = True
VertexAna.scores = 'offlineSlimmedPrimaryVertices'

from PhysicsTools.Heppy.analyzers.objects.PhotonAnalyzer import PhotonAnalyzer
PhoAna = PhotonAnalyzer.defaultConfig

from PhysicsTools.Heppy.analyzers.objects.TauAnalyzer import TauAnalyzer
TauAna = TauAnalyzer.defaultConfig
TauAna.inclusive_ptMin = 18.
TauAna.inclusive_etaMax = 2.5
TauAna.inclusive_dxyMax = 1000.
TauAna.inclusive_dzMax = 0.4
TauAna.inclusive_vetoLeptons = False
TauAna.inclusive_leptonVetoDR = 0.4
TauAna.inclusive_decayModeID = "decayModeFindingNewDMs"
TauAna.inclusive_tauID = "decayModeFindingNewDMs"
TauAna.inclusive_vetoLeptonsPOG = False
TauAna.inclusive_tauAntiMuonID = ""
TauAna.inclusive_tauAntiElectronID = ""
TauAna.inclusive_tauLooseID = "decayModeFindingNewDMs"

from PhysicsTools.Heppy.analyzers.objects.JetAnalyzer import JetAnalyzer
JetAna = JetAnalyzer.defaultConfig
JetAna.calculateSeparateCorrections = True # CV: needed for ttH prompt lepton MVA

from PhysicsTools.Heppy.analyzers.gen.LHEAnalyzer import LHEAnalyzer 
LHEAna = LHEAnalyzer.defaultConfig

from PhysicsTools.Heppy.analyzers.gen.LHEWeightAnalyzer import LHEWeightAnalyzer 
LHEWeightAna = LHEWeightAnalyzer.defaultConfig

from PhysicsTools.Heppy.analyzers.gen.GeneratorAnalyzer import GeneratorAnalyzer 
GenAna = GeneratorAnalyzer.defaultConfig

from VHbbAnalysis.Heppy.VHGeneratorAnalyzer import GeneratorAnalyzer as  VHGeneratorAnalyzer
VHGenAna = VHGeneratorAnalyzer.defaultConfig

from PhysicsTools.Heppy.analyzers.objects.METAnalyzer import METAnalyzer
METAna = METAnalyzer.defaultConfig
METAna.applyJetSmearing = False
METAna.doTkMet = True
METAna.doMetNoPU = False
METAna.doTkGenMet = False
METAna.includeTkMetPVLoose = False
METAna.includeTkMetPVTight = False

METPuppiAna = copy.copy(METAna)
METPuppiAna.metCollection     = "slimmedMETsPuppi"
METPuppiAna.doMetNoPU = False
METPuppiAna.recalibrate = False
METPuppiAna.collectionPostFix = "Puppi"
METPuppiAna.copyMETsByValue = True
METPuppiAna.doTkMet = False
METPuppiAna.doMetNoPU = False

from PhysicsTools.Heppy.analyzers.core.PileUpAnalyzer import PileUpAnalyzer
PUAna = PileUpAnalyzer.defaultConfig

from VHbbAnalysis.Heppy.VHbbAnalyzer import VHbbAnalyzer
JetAna.jetPt = 15
JetAna.jetEta = 4.7
JetAna.doQG=True
JetAna.QGpath=os.environ['CMSSW_BASE']+"/src/PhysicsTools/Heppy/data/pdfQG_AK4chs_13TeV_v2b.root"
JetAna.recalibrateJets=True
JetAna.jecPath=os.environ['CMSSW_BASE']+"/src/VHbbAnalysis/Heppy/data/jec"
#JetAna.mcGT="Fall15_25nsV2_MC"
#JetAna.dataGT = "Fall15_25nsV2_DATA"
JetAna.mcGT="Spring16_25nsV6_MC"
JetAna.dataGT="Spring16_25nsV6_DATA"
JetAna.addJECShifts=True
JetAna.addJERShifts=True

# delta-beta corrected isolation for muons:
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2#Muon_Isolation
def mu_pfRelIso04(mu):
	return (mu.pfIsolationR04().sumChargedHadronPt + max( mu.pfIsolationR04().sumNeutralHadronEt + mu.pfIsolationR04().sumPhotonEt - 0.5 * mu.pfIsolationR04().sumPUPt,0.0)) / mu.pt()

# the MVA-triggering preseletion according to:
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/MultivariateElectronIdentificationRun2#Recommended_MVA_recipes_for_2015
def ele_mvaEleID_Trig_preselection(ele) : 
	return (ele.pt()>15 and 
		( ( abs(ele.superCluster().eta()) < 1.4442 and ele.full5x5_sigmaIetaIeta() < 0.012 and ele.hcalOverEcal() < 0.09 and (ele.ecalPFClusterIso() / ele.pt()) < 0.37 and (ele.hcalPFClusterIso() / ele.pt()) < 0.25 and (ele.dr03TkSumPt() / ele.pt()) < 0.18 and abs(ele.deltaEtaSuperClusterTrackAtVtx()) < 0.0095 and abs(ele.deltaPhiSuperClusterTrackAtVtx()) < 0.065 ) or 
		  ( abs(ele.superCluster().eta()) > 1.5660 and ele.full5x5_sigmaIetaIeta() < 0.033 and ele.hcalOverEcal() <0.09 and (ele.ecalPFClusterIso() / ele.pt()) < 0.45 and (ele.hcalPFClusterIso() / ele.pt()) < 0.28 and (ele.dr03TkSumPt() / ele.pt()) < 0.18 ) ) )

from VHbbAnalysis.Heppy.ttHLeptonMVAAnalyzer import ttHLeptonMVAAnalyzer
ttHLeptonMVA = cfg.Analyzer(
    verbose = False,
    class_object = ttHLeptonMVAAnalyzer,
)

from VHbbAnalysis.Heppy.TriggerEmulation import TriggerEmulationAnalyzer
trigemu = cfg.Analyzer(
    verbose = False,
    class_object = TriggerEmulationAnalyzer,
    calibrationFile="triggerEmulation.root",
    slEleSelection = lambda x : x.pt() > 25 and getattr(x,"mvaIdSpring15TrigTight",False) and ele_mvaEleID_Trig_preselection(x),
    slMuSelection = lambda x : x.pt() > 25 and x.muonID("POG_ID_Tight") and mu_pfRelIso04(x) < 0.15,
    dlEleSelection = lambda x : x.pt() > 15 and getattr(x,"mvaIdSpring15TrigMedium",False) and ele_mvaEleID_Trig_preselection(x),
    dlMuSelection = lambda x : x.pt() > 15 and x.muonID("POG_ID_Loose") and mu_pfRelIso04(x) < 0.25,
)

VHbb = cfg.Analyzer(
    verbose=False,
    class_object=VHbbAnalyzer,
    #wEleSelection = lambda x : x.pt() > 25 and x.electronID("cutBasedElectronID-Spring15-25ns-V1-standalone-tight"),
    wEleSelection = lambda x : x.pt() > 25 and getattr(x,"mvaIdSpring15TrigTight",False) and ele_mvaEleID_Trig_preselection(x),
    #wMuSelection = lambda x : x.pt() > 25 and x.muonID("POG_ID_Tight"),
    wMuSelection = lambda x : x.pt() > 25 and x.muonID("POG_ID_Tight") and mu_pfRelIso04(x) < 0.15,
    #zEleSelection = lambda x : x.pt() > 15 and x.electronID("cutBasedElectronID-Spring15-25ns-V1-standalone-loose"),
    zEleSelection = lambda x : x.pt() > 15 and getattr(x,"mvaIdSpring15TrigMedium",False) and ele_mvaEleID_Trig_preselection(x),
    #zMuSelection = lambda x : x.pt() > 10 and x.muonID("POG_ID_Loose"),
    zMuSelection = lambda x : x.pt() > 15 and x.muonID("POG_ID_Loose") and mu_pfRelIso04(x) < 0.25,
    zLeadingElePt = 20,
    zLeadingMuPt = 20,
    singleJetThreshold = 200,
    sumPtThreshold = 160.,
    higgsJetsPreSelection = lambda x: (( x.puJetId() > 0 and x.jetID('POG_PFID_Loose')) or abs(x.eta())>3.0 ) and x.pt() >  20 ,
    higgsJetsPreSelectionVBF = lambda x: x.pt() >  20 ,
#    higgsJetsPreSelectionVBF = lambda x: (( x.puJetId() > 0 and x.jetID('POG_PFID_Loose')) or abs(x.eta())>3.0 ) and x.pt() >  20,
    passall=False,
    doSoftActivityVH=True,
    doVBF=True,
    regressions = [
        {"weight":"ttbar-spring16-80X.weights.xml", "name":"jet0Regression", "vtypes":[0,1,2,3,4,5,-1]},
    ],
    regressionVBF = [
        {"weight":"ttbar-spring16-80X.weights.xml", "name":"jet0Regression_vbf", "vtypes":[0,1,2,3,4,5,-1]}
    ],
    VBFblikelihood = {"weight":"TMVA_blikelihood_vbf_cmssw76_h21trained.weights.xml", "name":"BDGT"}
)

from VHbbAnalysis.Heppy.TTHtoTauTauAnalyzer import TTHtoTauTauAnalyzer
TTHtoTauTau = cfg.Analyzer(
    verbose = False,
    class_object = TTHtoTauTauAnalyzer,
)
from VHbbAnalysis.Heppy.TTHtoTauTauGeneratorAnalyzer import TTHtoTauTauGeneratorAnalyzer
TTHtoTauTauGen = cfg.Analyzer(
    verbose = False,
    class_object = TTHtoTauTauGeneratorAnalyzer,
)

#from VHbbAnalysis.Heppy.HeppyShell import HeppyShell
#sh = cfg.Analyzer( class_object=HeppyShell)

from PhysicsTools.Heppy.analyzers.core.TriggerBitAnalyzer import TriggerBitAnalyzer

from VHbbAnalysis.Heppy.TriggerTable import triggerTable
from VHbbAnalysis.Heppy.TriggerTableData import triggerTable as triggerTableData

TrigAna = cfg.Analyzer(
    verbose = False,
    class_object = TriggerBitAnalyzer,
    triggerBits = triggerTable,  #default is MC, use the triggerTableData in -data.py files
#   processName = 'HLT',
#   outprefix = 'HLT'
   )

from PhysicsTools.HeppyCore.framework.services.tfile import TFileService 
output_service = cfg.Service(
      TFileService,
      'outputfile',
      name="outputfile",
      fname='tree.root',
      option='recreate'
    )

from PhysicsTools.Heppy.analyzers.core.TriggerBitAnalyzer import TriggerBitAnalyzer
FlagsAna = TriggerBitAnalyzer.defaultEventFlagsConfig
FlagsAna.triggerBits.update( { "chargedHadronTrackResolutionFilter" : ["Flag_chargedHadronTrackResolutionFilter"], "muonBadTrackFilter" : ["Flag_muonBadTrackFilter"], "CSCTightHalo2015Filter" : ["Flag_CSCTightHalo2015Filter"] } )

from VHbbAnalysis.Heppy.hbheAnalyzer import *
hbheAna = hbheAnalyzer.defaultConfig


### add trigger objects ####
from PhysicsTools.Heppy.analyzers.core.TriggerObjectsAnalyzer import TriggerObjectsAnalyzer
from VHbbAnalysis.Heppy.TriggerObjectsList import *
TriggerObjectsAna = TriggerObjectsAnalyzer.defaultConfig
TriggerObjectsAna.triggerObjectsCfgs = triggerObjectCollections

for collectionName in triggerObjectCollectionsFull.keys():
    treeProducer.collections["trgObjects_"+collectionName] = NTupleCollection("trgObjects_"+collectionName, triggerObjectsType, 5, help="")

for collectionName in triggerObjectCollectionsOnlyPt.keys():
    treeProducer.collections["trgObjects_"+collectionName] = NTupleCollection("trgObjects_"+collectionName, triggerObjectsOnlyPtType, 5, help="")

for collectionName in triggerObjectCollectionsOnlySize.keys():
    treeProducer.collections["trgObjects_"+collectionName] = NTupleCollection("trgObjects_"+collectionName, triggerObjectsNothingType , 5, help="")
#    treeProducer.globalVariables.append(NTupleVariable("trgObjects_"+collectionName+"_size", lambda ev : len(getattr(ev,"trgObjects_"+collectionName,[])), int, help="trigger objects size"))

###


from PhysicsTools.Heppy.analyzers.gen.PDFWeightsAnalyzer import PDFWeightsAnalyzer
PdfAna = cfg.Analyzer(PDFWeightsAnalyzer,
    PDFWeights = [],
    doPDFVars = doPDFVars
)

if doPDFVars:
    treeProducer.globalVariables += [
        NTupleVariable("pdf_x1",  lambda ev: ev.pdf_x1, float,mcOnly=True, help="PDF energy fraction of first parton"),
        NTupleVariable("pdf_x2",  lambda ev: ev.pdf_x2, float,mcOnly=True, help="PDF energy fraction of second parton"),
        NTupleVariable("pdf_id1",  lambda ev: ev.pdf_id1, int,mcOnly=True, help="PDF id of first parton"),
        NTupleVariable("pdf_id2",  lambda ev: ev.pdf_id2, int,mcOnly=True, help="PDF id of second parton"),
        NTupleVariable("pdf_scale",  lambda ev: ev.pdf_scale, float,mcOnly=True, help="PDF scale"),
    ]

TrigAna.unrollbits=True

from PhysicsTools.Heppy.analyzers.core.JSONAnalyzer import JSONAnalyzer
jsonAna = cfg.Analyzer(JSONAnalyzer,
      passAll=True
      )
silverJsonAna = cfg.Analyzer(JSONAnalyzer,
      passAll=True,
      json="silver.txt",
      suffix="_silver"
      )

sequence = [
    jsonAna,LHEAna,LHEWeightAna,FlagsAna,
    hbheAna, GenAna,VHGenAna,PUAna,TrigAna,
    VertexAna,LepAna,PhoAna,TauAna,JetAna,
    ttHLeptonMVA,METAna, METPuppiAna,
    PdfAna,
    VHbb,TTHtoTauTau,TTHtoTauTauGen,TriggerObjectsAna,trigemu,treeProducer
]

from PhysicsTools.Heppy.utils.miniAodFiles import miniAodFiles
sample = cfg.MCComponent(
	files = [
		#"root://xrootd.ba.infn.it//store/mc/RunIIFall15MiniAODv1/TT_TuneCUETP8M1_13TeV-powheg-scaledown-pythia8/MINIAODSIM/PU25nsData2015v1_76X_mcRun2_asymptotic_v12-v1/30000/045996FE-A19D-E511-B76D-D4AE526A0B47.root" ##ttbar
		#"root://xrootd.ba.infn.it//store/mc/RunIISpring16MiniAODv1/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/0899BDA9-AE01-E611-A239-008CFA05EA2C.root"
		"root://stormgf1.pi.infn.it:1094//store/mc/RunIISpring16MiniAODv2/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_miniAODv2_v0_ext4-v1/00000/E8090432-8628-E611-8713-001EC9ADFDC9.root",
		#"root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/mc/RunIISpring16MiniAODv2/WW_TuneCUETP8M1_13TeV-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_miniAODv2_v0-v1/50000/0AF21AF1-121B-E611-B652-549F35AE4F88.root"
		],
    #files = ["226BB247-A565-E411-91CF-00266CFF0AF4.root"],
    name="ZHLL125", isEmbed=False,
    puFileMC="puMC.root",
    puFileData="puData.root", 
    puFileDataPlus="puDataPlus.root", 
    puFileDataMinus="puDataMinus.root", 
    splitFactor = 5
    )
sample.isMC=True


# the following is declared in case this cfg is used in input to the heppy.py script
selectedComponents = [sample]
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
config = cfg.Config( components = selectedComponents,
                     sequence = sequence, 
		     services = [output_service],
                     events_class = Events)

class TestFilter(logging.Filter):
    def filter(self, record):
        print record

# and the following runs the process directly 
if __name__ == '__main__':
    from PhysicsTools.HeppyCore.framework.looper import Looper 
    looper = Looper( 'Loop', config, nPrint = 1, nEvents = 1000)

    import time
    import cProfile
    p = cProfile.Profile(time.clock)
    p.runcall(looper.loop)
    p.print_stats()
    looper.write()
