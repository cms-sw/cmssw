from __future__ import print_function
from __future__ import absolute_import
from builtins import range
import ROOT, os, math, sys
import numpy as num
from DataFormats.FWLite import Events, Handle
from .DeltaR import *

from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Muon import Muon
from PhysicsTools.Heppy.physicsobjects.Electron import Electron
from PhysicsTools.Heppy.physicsobjects.Tau import Tau
from PhysicsTools.Heppy.physicsutils.TauDecayModes import tauDecayModes


ROOT.gROOT.SetBatch(True)

#RelVal = '7_6_0_pre7'
#RelVal = '7_6_0'
#RelVal = '7_6_1'
#RelVal = '7_6_1_v2'
RelVal = '7_6_1_v3'

tag = 'v11'
if RelVal=='7_6_0_pre7':
    tag = 'v5'

tauH = Handle('vector<pat::Tau>')
vertexH = Handle('std::vector<reco::Vertex>')
genParticlesH  = Handle ('std::vector<reco::GenParticle>')
jetH = Handle('vector<pat::Jet>')

filelist = []
argvs = sys.argv
argc = len(argvs)

if argc != 2:
    print('Please specify the runtype : python3 runTauDisplay.py <ZTT, ZEE, ZMM, QCD>')
    sys.exit(0)

runtype = argvs[1]

print('You selected', runtype)



#def isFinal(p):
#    return not (p.numberOfDaughters() == 1 and p.daughter(0).pdgId() == p.pdgId())


def returnRough(dm):
    if dm in [0]:
        return 0
    elif dm in [1,2]:
        return 1
    elif dm in [5,6]:
        return 2
    elif dm in [10]:
        return 3
    elif dm in [11]:
        return 4
    else:
        return -1


def finalDaughters(gen, daughters=None):
    if daughters is None:
        daughters = []
    for i in range(gen.numberOfDaughters()):
        daughter = gen.daughter(i)
        if daughter.numberOfDaughters() == 0:
            daughters.append(daughter)
        else:
            finalDaughters(daughter, daughters)

    return daughters


def visibleP4(gen):
    final_ds = finalDaughters(gen)

    p4 = sum((d.p4() for d in final_ds if abs(d.pdgId()) not in [12, 14, 16]), ROOT.math.XYZTLorentzVectorD())

    return p4





for ii in range(1, 4):

    if runtype == 'ZEE' and RelVal.find('7_6_1')!=-1: pass
    else:
        if ii==3 : continue

    filename = ''
    
    if runtype == 'ZTT':
        filename = 'root://eoscms//eos/cms/store/cmst3/user/ytakahas/TauPOG/ZTT_CMSSW_' + RelVal + '_RelValZTT_13_MINIAODSIM_76X_mcRun2_asymptotic_' + tag + '-v1_' + str(ii) + '.root'
    elif runtype == 'ZEE':
        filename = 'root://eoscms//eos/cms/store/cmst3/user/ytakahas/TauPOG/ZEE_CMSSW_' + RelVal + '_RelValZEE_13_MINIAODSIM_76X_mcRun2_asymptotic_' + tag + '-v1_' + str(ii) + '.root'
    elif runtype == 'ZMM':
        filename = 'root://eoscms//eos/cms/store/cmst3/user/ytakahas/TauPOG/ZMM_CMSSW_' + RelVal + '_RelValZMM_13_MINIAODSIM_76X_mcRun2_asymptotic_' + tag + '-v1_' + str(ii) + '.root'
    elif runtype == 'QCD':
        filename = 'root://eoscms//eos/cms/store/cmst3/user/ytakahas/TauPOG/QCD_CMSSW_' + RelVal + '_QCD_FlatPt_15_3000HS_13_MINIAODSIM_76X_mcRun2_asymptotic_v11-v1_' + str(ii) + '.root'

    print(filename)
    filelist.append(filename)

events = Events(filelist)
print(len(filelist), 'files will be analyzed')


outputname = 'Myroot_' + RelVal + '_' + runtype + '.root'
file = ROOT.TFile(outputname, 'recreate')

h_ngen = ROOT.TH1F("h_ngen", "h_ngen",10,0,10)

tau_tree = ROOT.TTree('per_tau','per_tau')

tau_eventid = num.zeros(1, dtype=int)
tau_id = num.zeros(1, dtype=int)
tau_dm = num.zeros(1, dtype=int)
tau_dm_rough = num.zeros(1, dtype=int)
tau_pt = num.zeros(1, dtype=float)
tau_eta = num.zeros(1, dtype=float)
tau_phi = num.zeros(1, dtype=float)
tau_mass = num.zeros(1, dtype=float)
tau_gendm = num.zeros(1, dtype=int)
tau_gendm_rough = num.zeros(1, dtype=int)
tau_genpt = num.zeros(1, dtype=float)
tau_geneta = num.zeros(1, dtype=float)
tau_genphi = num.zeros(1, dtype=float)
tau_vertex = num.zeros(1, dtype=int)

tau_againstMuonLoose3 = num.zeros(1, dtype=int)
tau_againstMuonTight3 = num.zeros(1, dtype=int)

tau_againstElectronVLooseMVA5  = num.zeros(1, dtype=int)
tau_againstElectronLooseMVA5 = num.zeros(1, dtype=int)
tau_againstElectronMediumMVA5 = num.zeros(1, dtype=int)
tau_againstElectronTightMVA5 = num.zeros(1, dtype=int)
tau_againstElectronVTightMVA5 = num.zeros(1, dtype=int)
tau_againstElectronMVA5raw = num.zeros(1, dtype=float)
tau_byIsolationMVA3oldDMwLTraw = num.zeros(1, dtype=float)
tau_byLooseIsolationMVA3oldDMwLT = num.zeros(1, dtype=int)
tau_byMediumIsolationMVA3oldDMwLT = num.zeros(1, dtype=int)
tau_byTightIsolationMVA3oldDMwLT = num.zeros(1, dtype=int)
tau_byVLooseIsolationMVA3oldDMwLT = num.zeros(1, dtype=int)
tau_byVTightIsolationMVA3oldDMwLT = num.zeros(1, dtype=int)
tau_byVVTightIsolationMVA3oldDMwLT = num.zeros(1, dtype=int)


tau_byCombinedIsolationDeltaBetaCorrRaw3Hits = num.zeros(1, dtype=float)
tau_byLooseCombinedIsolationDeltaBetaCorr3Hits = num.zeros(1, dtype=int)
tau_byMediumCombinedIsolationDeltaBetaCorr3Hits = num.zeros(1, dtype=int)
tau_byTightCombinedIsolationDeltaBetaCorr3Hits = num.zeros(1, dtype=int)
tau_chargedIsoPtSum = num.zeros(1, dtype=float)
tau_neutralIsoPtSum = num.zeros(1, dtype=float)
tau_puCorrPtSum = num.zeros(1, dtype=float)
tau_byLoosePileupWeightedIsolation3Hits = num.zeros(1, dtype=int)
tau_byMediumPileupWeightedIsolation3Hits = num.zeros(1, dtype=int)
tau_byTightPileupWeightedIsolation3Hits = num.zeros(1, dtype=int)
tau_byPileupWeightedIsolationRaw3Hits = num.zeros(1, dtype=float)
tau_neutralIsoPtSumWeight = num.zeros(1, dtype=float)
tau_footprintCorrection = num.zeros(1, dtype=float)
tau_photonPtSumOutsideSignalCone = num.zeros(1, dtype=float)
tau_decayModeFindingOldDMs = num.zeros(1, dtype=int)
tau_decayModeFindingNewDMs = num.zeros(1, dtype=int)


tau_againstElectronVLooseMVA6  = num.zeros(1, dtype=int)
tau_againstElectronLooseMVA6 = num.zeros(1, dtype=int)
tau_againstElectronMediumMVA6 = num.zeros(1, dtype=int)
tau_againstElectronTightMVA6 = num.zeros(1, dtype=int)
tau_againstElectronVTightMVA6 = num.zeros(1, dtype=int)
tau_againstElectronMVA6raw = num.zeros(1, dtype=float)

#'byIsolationMVArun2v1DBdR03oldDMwLTraw' 
tau_byIsolationMVArun2v1DBoldDMwLTraw = num.zeros(1, dtype=float)
tau_byVLooseIsolationMVArun2v1DBoldDMwLT = num.zeros(1, dtype=int)
tau_byLooseIsolationMVArun2v1DBoldDMwLT  = num.zeros(1, dtype=int)
tau_byMediumIsolationMVArun2v1DBoldDMwLT = num.zeros(1, dtype=int)
tau_byTightIsolationMVArun2v1DBoldDMwLT  = num.zeros(1, dtype=int)
tau_byVTightIsolationMVArun2v1DBoldDMwLT = num.zeros(1, dtype=int)
tau_byVVTightIsolationMVArun2v1DBoldDMwLT = num.zeros(1, dtype=int)

#byIsolationMVArun2v1PWdR03oldDMwLTraw'
#'byLooseCombinedIsolationDeltaBetaCorr3HitsdR03'  
#'byLooseIsolationMVArun2v1DBdR03oldDMwLT' 
#'byLooseIsolationMVArun2v1PWdR03oldDMwLT' 
#'byMediumCombinedIsolationDeltaBetaCorr3HitsdR03' 
#'byMediumIsolationMVArun2v1DBdR03oldDMwLT' 
#'byMediumIsolationMVArun2v1PWdR03oldDMwLT' 
#'byTightCombinedIsolationDeltaBetaCorr3HitsdR03' 
#'byTightIsolationMVArun2v1DBdR03oldDMwLT' 
#'byTightIsolationMVArun2v1PWdR03oldDMwLT' 
tau_byIsolationMVArun2v1PWoldDMwLTraw = num.zeros(1, dtype=float)
tau_byLooseIsolationMVArun2v1PWoldDMwLT = num.zeros(1, dtype=int)
tau_byMediumIsolationMVArun2v1PWoldDMwLT = num.zeros(1, dtype=int)
tau_byTightIsolationMVArun2v1PWoldDMwLT = num.zeros(1, dtype=int)
tau_byVLooseIsolationMVArun2v1PWoldDMwLT = num.zeros(1, dtype=int)
tau_byVTightIsolationMVArun2v1PWoldDMwLT = num.zeros(1, dtype=int)
tau_byVVTightIsolationMVArun2v1PWoldDMwLT = num.zeros(1, dtype=int)

#'byVLooseIsolationMVArun2v1DBdR03oldDMwLT' 
#'byVLooseIsolationMVArun2v1PWdR03oldDMwLT' 
#'byVTightIsolationMVArun2v1DBdR03oldDMwLT' 
#'byVTightIsolationMVArun2v1PWdR03oldDMwLT' 
#'byVVTightIsolationMVArun2v1DBdR03oldDMwLT' 
#'byVVTightIsolationMVArun2v1PWdR03oldDMwLT' 



tau_tree.Branch('tau_id', tau_id, 'tau_id/I')
tau_tree.Branch('tau_vertex', tau_vertex, 'tau_vertex/I')
tau_tree.Branch('tau_eventid', tau_eventid, 'tau_eventid/I')
tau_tree.Branch('tau_dm', tau_dm, 'tau_dm/I')
tau_tree.Branch('tau_dm_rough', tau_dm_rough, 'tau_dm_rough/I')
tau_tree.Branch('tau_pt', tau_pt, 'tau_pt/D')
tau_tree.Branch('tau_eta', tau_eta, 'tau_eta/D')
tau_tree.Branch('tau_phi', tau_phi, 'tau_phi/D')
tau_tree.Branch('tau_mass', tau_mass, 'tau_mass/D')
tau_tree.Branch('tau_gendm', tau_gendm, 'tau_gendm/I')
tau_tree.Branch('tau_gendm_rough', tau_gendm_rough, 'tau_gendm_rough/I')
tau_tree.Branch('tau_genpt', tau_genpt, 'tau_genpt/D')
tau_tree.Branch('tau_geneta', tau_geneta, 'tau_geneta/D')
tau_tree.Branch('tau_genphi', tau_genphi, 'tau_genphi/D')

tau_tree.Branch('tau_againstMuonLoose3', tau_againstMuonLoose3, 'tau_againstMuonLoose3/I')
tau_tree.Branch('tau_againstMuonTight3', tau_againstMuonTight3, 'tau_againstMuonTight3/I')

tau_tree.Branch('tau_againstElectronVLooseMVA5', tau_againstElectronVLooseMVA5, 'tau_againstElectronVLooseMVA5/I')
tau_tree.Branch('tau_againstElectronLooseMVA5', tau_againstElectronLooseMVA5, 'tau_againstElectronLooseMVA5/I')
tau_tree.Branch('tau_againstElectronMediumMVA5', tau_againstElectronMediumMVA5, 'tau_againstElectronMediumMVA5/I')
tau_tree.Branch('tau_againstElectronTightMVA5', tau_againstElectronTightMVA5, 'tau_againstElectronTightMVA5/I')
tau_tree.Branch('tau_againstElectronVTightMVA5', tau_againstElectronVTightMVA5, 'tau_againstElectronVTightMVA5/I')
tau_tree.Branch('tau_againstElectronMVA5raw', tau_againstElectronMVA5raw, 'tau_againstElectronMVA5raw/D')

tau_tree.Branch('tau_againstElectronVLooseMVA6', tau_againstElectronVLooseMVA6, 'tau_againstElectronVLooseMVA6/I')
tau_tree.Branch('tau_againstElectronLooseMVA6', tau_againstElectronLooseMVA6, 'tau_againstElectronLooseMVA6/I')
tau_tree.Branch('tau_againstElectronMediumMVA6', tau_againstElectronMediumMVA6, 'tau_againstElectronMediumMVA6/I')
tau_tree.Branch('tau_againstElectronTightMVA6', tau_againstElectronTightMVA6, 'tau_againstElectronTightMVA6/I')
tau_tree.Branch('tau_againstElectronVTightMVA6', tau_againstElectronVTightMVA6, 'tau_againstElectronVTightMVA6/I')
tau_tree.Branch('tau_againstElectronMVA6raw', tau_againstElectronMVA6raw, 'tau_againstElectronMVA6raw/D')


tau_tree.Branch('tau_byCombinedIsolationDeltaBetaCorrRaw3Hits', tau_byCombinedIsolationDeltaBetaCorrRaw3Hits, 'tau_byCombinedIsolationDeltaBetaCorrRaw3Hits/D')
tau_tree.Branch('tau_byLooseCombinedIsolationDeltaBetaCorr3Hits', tau_byLooseCombinedIsolationDeltaBetaCorr3Hits, 'tau_byLooseCombinedIsolationDeltaBetaCorr3Hits/I')
tau_tree.Branch('tau_byMediumCombinedIsolationDeltaBetaCorr3Hits', tau_byMediumCombinedIsolationDeltaBetaCorr3Hits, 'tau_byMediumCombinedIsolationDeltaBetaCorr3Hits/I')
tau_tree.Branch('tau_byTightCombinedIsolationDeltaBetaCorr3Hits', tau_byTightCombinedIsolationDeltaBetaCorr3Hits, 'tau_byTightCombinedIsolationDeltaBetaCorr3Hits/I')
tau_tree.Branch('tau_chargedIsoPtSum', tau_chargedIsoPtSum, 'tau_chargedIsoPtSum/D')
tau_tree.Branch('tau_neutralIsoPtSum', tau_neutralIsoPtSum, 'tau_neutralIsoPtSum/D')
tau_tree.Branch('tau_puCorrPtSum', tau_puCorrPtSum, 'tau_puCorrPtSum/D')
tau_tree.Branch('tau_byLoosePileupWeightedIsolation3Hits', tau_byLoosePileupWeightedIsolation3Hits, 'tau_byLoosePileupWeightedIsolation3Hits/I')
tau_tree.Branch('tau_byMediumPileupWeightedIsolation3Hits', tau_byMediumPileupWeightedIsolation3Hits, 'tau_byMediumPileupWeightedIsolation3Hits/I')
tau_tree.Branch('tau_byTightPileupWeightedIsolation3Hits', tau_byTightPileupWeightedIsolation3Hits, 'tau_byTightPileupWeightedIsolation3Hits/I')
tau_tree.Branch('tau_byPileupWeightedIsolationRaw3Hits', tau_byPileupWeightedIsolationRaw3Hits, 'tau_byPileupWeightedIsolationRaw3Hits/D')
tau_tree.Branch('tau_neutralIsoPtSumWeight', tau_neutralIsoPtSumWeight, 'tau_neutralIsoPtSumWeight/D')
tau_tree.Branch('tau_footprintCorrection', tau_footprintCorrection, 'tau_footprintCorrection/D')
tau_tree.Branch('tau_photonPtSumOutsideSignalCone', tau_photonPtSumOutsideSignalCone, 'tau_photonPtSumOutsideSignalCone/D')
tau_tree.Branch('tau_decayModeFindingOldDMs', tau_decayModeFindingOldDMs, 'tau_decayModeFindingOldDMs/I')
tau_tree.Branch('tau_decayModeFindingNewDMs', tau_decayModeFindingNewDMs, 'tau_decayModeFindingNewDMs/I')

tau_tree.Branch('tau_byIsolationMVA3oldDMwLTraw', tau_byIsolationMVA3oldDMwLTraw, 'tau_byIsolationMVA3oldDMwLTraw/D')
tau_tree.Branch('tau_byLooseIsolationMVA3oldDMwLT', tau_byLooseIsolationMVA3oldDMwLT, 'tau_byLooseIsolationMVA3oldDMwLT/I')
tau_tree.Branch('tau_byMediumIsolationMVA3oldDMwLT', tau_byMediumIsolationMVA3oldDMwLT, 'tau_byMediumIsolationMVA3oldDMwLT/I')
tau_tree.Branch('tau_byTightIsolationMVA3oldDMwLT', tau_byTightIsolationMVA3oldDMwLT, 'tau_byTightIsolationMVA3oldDMwLT/I')
tau_tree.Branch('tau_byVLooseIsolationMVA3oldDMwLT', tau_byVLooseIsolationMVA3oldDMwLT, 'tau_byVLooseIsolationMVA3oldDMwLT/I')
tau_tree.Branch('tau_byVTightIsolationMVA3oldDMwLT', tau_byVTightIsolationMVA3oldDMwLT, 'tau_byVTightIsolationMVA3oldDMwLT/I')
tau_tree.Branch('tau_byVVTightIsolationMVA3oldDMwLT', tau_byVVTightIsolationMVA3oldDMwLT, 'tau_byVVTightIsolationMVA3oldDMwLT/I')


tau_tree.Branch('tau_byIsolationMVArun2v1DBoldDMwLTraw', tau_byIsolationMVArun2v1DBoldDMwLTraw, 'tau_byIsolationMVArun2v1DBoldDMwLTraw/D')
tau_tree.Branch('tau_byLooseIsolationMVArun2v1DBoldDMwLT', tau_byLooseIsolationMVArun2v1DBoldDMwLT, 'tau_byLooseIsolationMVArun2v1DBoldDMwLT/I')
tau_tree.Branch('tau_byMediumIsolationMVArun2v1DBoldDMwLT', tau_byMediumIsolationMVArun2v1DBoldDMwLT, 'tau_byMediumIsolationMVArun2v1DBoldDMwLT/I')
tau_tree.Branch('tau_byTightIsolationMVArun2v1DBoldDMwLT', tau_byTightIsolationMVArun2v1DBoldDMwLT, 'tau_byTightIsolationMVArun2v1DBoldDMwLT/I')
tau_tree.Branch('tau_byVLooseIsolationMVArun2v1DBoldDMwLT', tau_byVLooseIsolationMVArun2v1DBoldDMwLT, 'tau_byVLooseIsolationMVArun2v1DBoldDMwLT/I')
tau_tree.Branch('tau_byVTightIsolationMVArun2v1DBoldDMwLT', tau_byVTightIsolationMVArun2v1DBoldDMwLT, 'tau_byVTightIsolationMVArun2v1DBoldDMwLT/I')
tau_tree.Branch('tau_byVVTightIsolationMVArun2v1DBoldDMwLT', tau_byVVTightIsolationMVArun2v1DBoldDMwLT, 'tau_byVVTightIsolationMVArun2v1DBoldDMwLT/I')

tau_tree.Branch('tau_byIsolationMVArun2v1PWoldDMwLTraw', tau_byIsolationMVArun2v1PWoldDMwLTraw, 'tau_byIsolationMVArun2v1PWoldDMwLTraw/D')
tau_tree.Branch('tau_byLooseIsolationMVArun2v1PWoldDMwLT', tau_byLooseIsolationMVArun2v1PWoldDMwLT, 'tau_byLooseIsolationMVArun2v1PWoldDMwLT/I')
tau_tree.Branch('tau_byMediumIsolationMVArun2v1PWoldDMwLT', tau_byMediumIsolationMVArun2v1PWoldDMwLT, 'tau_byMediumIsolationMVArun2v1PWoldDMwLT/I')
tau_tree.Branch('tau_byTightIsolationMVArun2v1PWoldDMwLT', tau_byTightIsolationMVArun2v1PWoldDMwLT, 'tau_byTightIsolationMVArun2v1PWoldDMwLT/I')
tau_tree.Branch('tau_byVLooseIsolationMVArun2v1PWoldDMwLT', tau_byVLooseIsolationMVArun2v1PWoldDMwLT, 'tau_byVLooseIsolationMVArun2v1PWoldDMwLT/I')
tau_tree.Branch('tau_byVTightIsolationMVArun2v1PWoldDMwLT', tau_byVTightIsolationMVArun2v1PWoldDMwLT, 'tau_byVTightIsolationMVArun2v1PWoldDMwLT/I')
tau_tree.Branch('tau_byVVTightIsolationMVArun2v1PWoldDMwLT', tau_byVVTightIsolationMVArun2v1PWoldDMwLT, 'tau_byVVTightIsolationMVArun2v1PWoldDMwLT/I')




evtid = 0

for event in events:
    
    evtid += 1  
    eid = event.eventAuxiliary().id().event()
    
    if evtid%1000 == 0:
        print('Event ', evtid, 'processed')

    event.getByLabel("slimmedTaus", tauH)
    event.getByLabel("offlineSlimmedPrimaryVertices", vertexH)
    event.getByLabel('prunedGenParticles',genParticlesH)
    event.getByLabel("slimmedJets", jetH)

    taus = tauH.product()
    vertices = vertexH.product()
    genParticles = genParticlesH.product()
    jets = [jet for jet in jetH.product() if jet.pt() > 20 and abs(jet.eta()) < 2.3]

    genTaus = [p for p in genParticles if abs(p.pdgId()) == 15 and p.status()==2]
    genElectrons = [p for p in genParticles if abs(p.pdgId()) == 11 and p.status()==1 and p.pt() > 20 and abs(p.eta())<2.3]
    genMuons = [p for p in genParticles if abs(p.pdgId()) == 13 and p.status()==1 and p.pt() > 20 and abs(p.eta())<2.3]

    for tau in taus:

        _genparticle_ = []

        for igen in genTaus:
                        
            visP4 = visibleP4(igen)

            gen_dm = tauDecayModes.genDecayModeInt([d for d in finalDaughters(igen) if abs(d.pdgId()) not in [12, 14, 16]])

            igen.decaymode = gen_dm
            igen.vis = visP4
            
            if abs(visP4.eta()) > 2.3: continue
            if visP4.pt() < 20: continue

            dr = deltaR(tau.eta(), tau.phi(), visP4.eta(), visP4.phi())

            if dr < 0.5:
                _genparticle_.append(igen)



        if runtype=='ZTT':
            h_ngen.Fill(len(genTaus))
            if len(_genparticle_) != 1: continue

        bmjet, _dr_ = bestMatch(tau, jets)
        if runtype=='QCD':
            h_ngen.Fill(len(jets))
            if bmjet == None: continue

        bme, _dr_ = bestMatch(tau, genElectrons)
        if runtype=='ZEE':
            h_ngen.Fill(len(genElectrons))
            if bme == None: continue

        bmm, _dr_ = bestMatch(tau, genMuons)
        if runtype=='ZMM':
            h_ngen.Fill(len(genMuons))
            if bmm == None: continue


        tau_id[0] = evtid
        tau_eventid[0] = eid
        tau_dm[0] = tau.decayMode()
        tau_dm_rough[0] = returnRough(tau.decayMode())
        tau_pt[0] = tau.pt()
        tau_eta[0] = tau.eta()
        tau_phi[0] = tau.phi()
        tau_mass[0] = tau.mass()
        tau_vertex[0] = len(vertices)

        tau_againstMuonLoose3[0] = tau.tauID('againstMuonLoose3')
        tau_againstMuonTight3[0] = tau.tauID('againstMuonTight3')

        tau_againstElectronVLooseMVA5[0] = tau.tauID('againstElectronVLooseMVA5')
        tau_againstElectronLooseMVA5[0] = tau.tauID('againstElectronLooseMVA5')
        tau_againstElectronMediumMVA5[0] = tau.tauID('againstElectronMediumMVA5')
        tau_againstElectronTightMVA5[0] = tau.tauID('againstElectronTightMVA5')
        tau_againstElectronVTightMVA5[0] = tau.tauID('againstElectronVTightMVA5')
        tau_againstElectronMVA5raw[0] = tau.tauID('againstElectronMVA5raw')

        tau_byCombinedIsolationDeltaBetaCorrRaw3Hits[0] = tau.tauID('byCombinedIsolationDeltaBetaCorrRaw3Hits')
        tau_byLooseCombinedIsolationDeltaBetaCorr3Hits[0] = tau.tauID('byLooseCombinedIsolationDeltaBetaCorr3Hits')
        tau_byMediumCombinedIsolationDeltaBetaCorr3Hits[0] = tau.tauID('byMediumCombinedIsolationDeltaBetaCorr3Hits')
        tau_byTightCombinedIsolationDeltaBetaCorr3Hits[0] = tau.tauID('byTightCombinedIsolationDeltaBetaCorr3Hits')
        tau_chargedIsoPtSum[0] = tau.tauID('chargedIsoPtSum')
        tau_neutralIsoPtSum[0] = tau.tauID('neutralIsoPtSum')
        tau_puCorrPtSum[0] = tau.tauID('puCorrPtSum')
        tau_byLoosePileupWeightedIsolation3Hits[0] = tau.tauID('byLoosePileupWeightedIsolation3Hits')
        tau_byMediumPileupWeightedIsolation3Hits[0] = tau.tauID('byMediumPileupWeightedIsolation3Hits')
        tau_byTightPileupWeightedIsolation3Hits[0] = tau.tauID('byTightPileupWeightedIsolation3Hits')
        tau_byPileupWeightedIsolationRaw3Hits[0] = tau.tauID('byPileupWeightedIsolationRaw3Hits')
        tau_neutralIsoPtSumWeight[0] = tau.tauID('neutralIsoPtSumWeight')
        tau_footprintCorrection[0] = tau.tauID('footprintCorrection')
        tau_photonPtSumOutsideSignalCone[0] = tau.tauID('photonPtSumOutsideSignalCone')
        tau_decayModeFindingOldDMs[0] = tau.tauID('decayModeFinding')
        tau_decayModeFindingNewDMs[0] = tau.tauID('decayModeFindingNewDMs')

        tau_byIsolationMVA3oldDMwLTraw[0] = tau.tauID('byIsolationMVA3oldDMwLTraw')
        tau_byLooseIsolationMVA3oldDMwLT[0] = tau.tauID('byLooseIsolationMVA3oldDMwLT')
        tau_byMediumIsolationMVA3oldDMwLT[0] = tau.tauID('byMediumIsolationMVA3oldDMwLT')
        tau_byTightIsolationMVA3oldDMwLT[0] = tau.tauID('byTightIsolationMVA3oldDMwLT')
        tau_byVLooseIsolationMVA3oldDMwLT[0] = tau.tauID('byVLooseIsolationMVA3oldDMwLT')
        tau_byVTightIsolationMVA3oldDMwLT[0] = tau.tauID('byVTightIsolationMVA3oldDMwLT')
        tau_byVVTightIsolationMVA3oldDMwLT[0] = tau.tauID('byVVTightIsolationMVA3oldDMwLT')

        if RelVal.find('7_6_1')!=-1:
            tau_againstElectronVLooseMVA6[0] = tau.tauID('againstElectronVLooseMVA6')
            tau_againstElectronLooseMVA6[0] = tau.tauID('againstElectronLooseMVA6')
            tau_againstElectronMediumMVA6[0] = tau.tauID('againstElectronMediumMVA6')
            tau_againstElectronTightMVA6[0] = tau.tauID('againstElectronTightMVA6')
            tau_againstElectronVTightMVA6[0] = tau.tauID('againstElectronVTightMVA6')
            tau_againstElectronMVA6raw[0] = tau.tauID('againstElectronMVA6raw')

            tau_byIsolationMVArun2v1DBoldDMwLTraw[0] = tau.tauID('byIsolationMVArun2v1DBoldDMwLTraw')
            tau_byLooseIsolationMVArun2v1DBoldDMwLT[0] = tau.tauID('byLooseIsolationMVArun2v1DBoldDMwLT')
            tau_byMediumIsolationMVArun2v1DBoldDMwLT[0] = tau.tauID('byMediumIsolationMVArun2v1DBoldDMwLT')
            tau_byTightIsolationMVArun2v1DBoldDMwLT[0] = tau.tauID('byTightIsolationMVArun2v1DBoldDMwLT')
            tau_byVLooseIsolationMVArun2v1DBoldDMwLT[0] = tau.tauID('byVLooseIsolationMVArun2v1DBoldDMwLT')
            tau_byVTightIsolationMVArun2v1DBoldDMwLT[0] = tau.tauID('byVTightIsolationMVArun2v1DBoldDMwLT')
            tau_byVVTightIsolationMVArun2v1DBoldDMwLT[0] = tau.tauID('byVVTightIsolationMVArun2v1DBoldDMwLT')

            tau_byIsolationMVArun2v1PWoldDMwLTraw[0] = tau.tauID('byIsolationMVArun2v1PWoldDMwLTraw')
            tau_byLooseIsolationMVArun2v1PWoldDMwLT[0] = tau.tauID('byLooseIsolationMVArun2v1PWoldDMwLT')
            tau_byMediumIsolationMVArun2v1PWoldDMwLT[0] = tau.tauID('byMediumIsolationMVArun2v1PWoldDMwLT')
            tau_byTightIsolationMVArun2v1PWoldDMwLT[0] = tau.tauID('byTightIsolationMVArun2v1PWoldDMwLT')
            tau_byVLooseIsolationMVArun2v1PWoldDMwLT[0] = tau.tauID('byVLooseIsolationMVArun2v1PWoldDMwLT')
            tau_byVTightIsolationMVArun2v1PWoldDMwLT[0] = tau.tauID('byVTightIsolationMVArun2v1PWoldDMwLT')
            tau_byVVTightIsolationMVArun2v1PWoldDMwLT[0] = tau.tauID('byVVTightIsolationMVArun2v1PWoldDMwLT')


        if runtype == 'ZTT':
            gp = _genparticle_[0]
            tau_gendm[0] = gp.decaymode
            tau_gendm_rough[0] = returnRough(gp.decaymode)
            tau_genpt[0] = gp.vis.pt()
            tau_geneta[0] = gp.vis.eta()
            tau_genphi[0] = gp.vis.phi()
        elif runtype == 'QCD':
            tau_gendm[0] = -1
            tau_genpt[0] = bmjet.pt()
            tau_geneta[0] = bmjet.eta()
            tau_genphi[0] = bmjet.phi()
        elif runtype == 'ZEE':
            tau_gendm[0] = -1
            tau_genpt[0] = bme.pt()
            tau_geneta[0] = bme.eta()
            tau_genphi[0] = bme.phi()
        elif runtype == 'ZMM':
            tau_gendm[0] = -1
            tau_genpt[0] = bmm.pt()
            tau_geneta[0] = bmm.eta()
            tau_genphi[0] = bmm.phi()

        tau_tree.Fill()



print(evtid, 'events are processed !')

file.Write()
file.Close()

