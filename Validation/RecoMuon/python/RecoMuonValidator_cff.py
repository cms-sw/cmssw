import FWCore.ParameterSet.Config as cms

#####################################################################################
# Configurations for RecoMuonValidator
#

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoMuonValidator_cfi import *
#
from SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi import *
from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters

#tracker
muonAssociatorByHitsNoSimHitsHelperTrk = muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperTrk.UseTracker = True
muonAssociatorByHitsNoSimHitsHelperTrk.UseMuon  = False
recoMuonVMuAssoc_trk = recoMuonValidator.clone()
recoMuonVMuAssoc_trk.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Trk'
recoMuonVMuAssoc_trk.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperTrk'
recoMuonVMuAssoc_trk.trackType = 'inner'
recoMuonVMuAssoc_trk.selection = "isTrackerMuon"
recoMuonVMuAssoc_trk.simLabel = ("TPmu")
recoMuonVMuAssoc_trk.tpRefVector = True
recoMuonVMuAssoc_trk.tpSelector.src = ("TPmu")

#standalone
muonAssociatorByHitsNoSimHitsHelperStandalone = muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperStandalone.UseTracker = False
muonAssociatorByHitsNoSimHitsHelperStandalone.UseMuon  = True
recoMuonVMuAssoc_sta = recoMuonValidator.clone()
recoMuonVMuAssoc_sta.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Sta'
recoMuonVMuAssoc_sta.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperStandalone'
recoMuonVMuAssoc_sta.trackType = 'outer'
recoMuonVMuAssoc_sta.selection = "isStandAloneMuon"
recoMuonVMuAssoc_sta.simLabel = ("TPmu")
recoMuonVMuAssoc_sta.tpRefVector = True
recoMuonVMuAssoc_sta.tpSelector.src = ("TPmu")

#global
muonAssociatorByHitsNoSimHitsHelperGlobal = muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperGlobal.UseTracker = True
muonAssociatorByHitsNoSimHitsHelperGlobal.UseMuon  = True
recoMuonVMuAssoc_glb = recoMuonValidator.clone()
recoMuonVMuAssoc_glb.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Glb'
recoMuonVMuAssoc_glb.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperGlobal'
recoMuonVMuAssoc_glb.trackType = 'global'
recoMuonVMuAssoc_glb.selection = "isGlobalMuon"
recoMuonVMuAssoc_glb.simLabel = ("TPmu")
recoMuonVMuAssoc_glb.tpRefVector = True
recoMuonVMuAssoc_glb.tpSelector.src = ("TPmu")

#tight
muonAssociatorByHitsNoSimHitsHelperTight = muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperTight.UseTracker = True
muonAssociatorByHitsNoSimHitsHelperTight.UseMuon  = True
recoMuonVMuAssoc_tgt = recoMuonValidator.clone()
recoMuonVMuAssoc_tgt.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Tgt'
recoMuonVMuAssoc_tgt.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperTight'
recoMuonVMuAssoc_tgt.trackType = 'global'
recoMuonVMuAssoc_tgt.selection = 'isGlobalMuon'
recoMuonVMuAssoc_tgt.wantTightMuon = True
recoMuonVMuAssoc_tgt.beamSpot = 'offlineBeamSpot'
recoMuonVMuAssoc_tgt.primaryVertex = 'offlinePrimaryVertices'
recoMuonVMuAssoc_tgt.simLabel = ("TPmu")
recoMuonVMuAssoc_tgt.tpRefVector = True
recoMuonVMuAssoc_tgt.tpSelector.src = ("TPmu")

##########################################################################
# Muon validation sequence using RecoMuonValidator
#

muonValidationRMV_seq = cms.Sequence(
    muonAssociatorByHitsNoSimHitsHelperTrk +recoMuonVMuAssoc_trk
    +muonAssociatorByHitsNoSimHitsHelperStandalone +recoMuonVMuAssoc_sta
    +muonAssociatorByHitsNoSimHitsHelperGlobal +recoMuonVMuAssoc_glb
    +muonAssociatorByHitsNoSimHitsHelperTight +recoMuonVMuAssoc_tgt
    )

# not used
#
#tracker and PF
#muonAssociatorByHitsNoSimHitsHelperTrkPF = muonAssociatorByHitsNoSimHitsHelper.clone()
#muonAssociatorByHitsNoSimHitsHelperTrkPF.UseTracker = True
#muonAssociatorByHitsNoSimHitsHelperTrkPF.UseMuon  = False
#recoMuonVMuAssoc_trkPF = recoMuonValidator.clone()
#recoMuonVMuAssoc_trkPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_TrkPF'
#recoMuonVMuAssoc_trkPF.usePFMuon = True
#recoMuonVMuAssoc_trkPF.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperTrkPF'
#recoMuonVMuAssoc_trkPF.trackType = 'inner'
#recoMuonVMuAssoc_trkPF.selection = "isTrackerMuon & isPFMuon"
#
#seed of StandAlone
#muonAssociatorByHitsNoSimHitsHelperSeedStandalone = muonAssociatorByHitsNoSimHitsHelper.clone()
#muonAssociatorByHitsNoSimHitsHelperSeedStandalone.UseTracker = False
#muonAssociatorByHitsNoSimHitsHelperSeedStandalone.UseMuon  = True
#recoMuonVMuAssoc_seedSta = recoMuonValidator.clone()
#recoMuonVMuAssoc_seedSta.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_SeedSta'
#recoMuonVMuAssoc_seedSta.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperStandalone'
#recoMuonVMuAssoc_seedSta.trackType = 'outer'
#recoMuonVMuAssoc_seedSta.selection = ""
#
#standalone and PF
#muonAssociatorByHitsNoSimHitsHelperStandalonePF = muonAssociatorByHitsNoSimHitsHelper.clone()
#muonAssociatorByHitsNoSimHitsHelperStandalonePF.UseTracker = False
#muonAssociatorByHitsNoSimHitsHelperStandalonePF.UseMuon  = True
#recoMuonVMuAssoc_staPF = recoMuonValidator.clone()
#recoMuonVMuAssoc_staPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_StaPF'
#recoMuonVMuAssoc_staPF.usePFMuon = True
#recoMuonVMuAssoc_staPF.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperStandalonePF'
#recoMuonVMuAssoc_staPF.trackType = 'outer'
#recoMuonVMuAssoc_staPF.selection = "isStandAloneMuon & isPFMuon"
#
#global and PF
#muonAssociatorByHitsNoSimHitsHelperGlobalPF = muonAssociatorByHitsNoSimHitsHelper.clone()
#muonAssociatorByHitsNoSimHitsHelperGlobalPF.UseTracker = True
#muonAssociatorByHitsNoSimHitsHelperGlobalPF.UseMuon  = True
#recoMuonVMuAssoc_glbPF = recoMuonValidator.clone()
#recoMuonVMuAssoc_glbPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_GlbPF'
#recoMuonVMuAssoc_glbPF.usePFMuon = True
#recoMuonVMuAssoc_glbPF.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperGlobalPF'
#recoMuonVMuAssoc_glbPF.trackType = 'global'
#recoMuonVMuAssoc_glbPF.selection = "isGlobalMuon & isPFMuon"
#
#muonValidationRMV_seq = cms.Sequence(
#    muonAssociatorByHitsNoSimHitsHelperTrk +recoMuonVMuAssoc_trk
#    +muonAssociatorByHitsNoSimHitsHelperStandalone +recoMuonVMuAssoc_sta
#    +muonAssociatorByHitsNoSimHitsHelperGlobal +recoMuonVMuAssoc_glb
#    +muonAssociatorByHitsNoSimHitsHelperTight +recoMuonVMuAssoc_tgt
#     
#    +muonAssociatorByHitsNoSimHitsHelperTrkPF +recoMuonVMuAssoc_trkPF
#    +muonAssociatorByHitsNoSimHitsHelperStandalonePF +recoMuonVMuAssoc_staPF
#    +muonAssociatorByHitsNoSimHitsHelperGlobalPF +recoMuonVMuAssoc_glbPF
#    )
#
