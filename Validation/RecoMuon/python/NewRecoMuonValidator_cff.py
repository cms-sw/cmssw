import FWCore.ParameterSet.Config as cms

#####################################################################################
# Configurations for RecoMuonValidator
#

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.NewRecoMuonValidator_cfi import *
#
##import SimGeneral.MixingModule.mixNoPU_cfi
from SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi import *
from SimMuon.MCTruth.NewMuonAssociatorByHits_cfi import NewMuonAssociatorByHitsCommonParameters

#tracker
NEWmuonAssociatorByHitsNoSimHitsHelperTrk = muonAssociatorByHitsNoSimHitsHelper.clone()
NEWmuonAssociatorByHitsNoSimHitsHelperTrk.UseTracker = True
NEWmuonAssociatorByHitsNoSimHitsHelperTrk.UseMuon  = False
NEWrecoMuonVMuAssoc_trk = NewRecoMuonValidator.clone()
NEWrecoMuonVMuAssoc_trk.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Trk'
NEWrecoMuonVMuAssoc_trk.muAssocLabel = 'NEWmuonAssociatorByHitsNoSimHitsHelperTrk'
NEWrecoMuonVMuAssoc_trk.trackType = 'inner'
NEWrecoMuonVMuAssoc_trk.selection = "isTrackerMuon"

#tracker and PF
NEWmuonAssociatorByHitsNoSimHitsHelperTrkPF = muonAssociatorByHitsNoSimHitsHelper.clone()
NEWmuonAssociatorByHitsNoSimHitsHelperTrkPF.UseTracker = True
NEWmuonAssociatorByHitsNoSimHitsHelperTrkPF.UseMuon  = False
NEWrecoMuonVMuAssoc_trkPF = NewRecoMuonValidator.clone()
NEWrecoMuonVMuAssoc_trkPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_TrkPF'
NEWrecoMuonVMuAssoc_trkPF.usePFMuon = True
NEWrecoMuonVMuAssoc_trkPF.muAssocLabel = 'NEWmuonAssociatorByHitsNoSimHitsHelperTrkPF'
NEWrecoMuonVMuAssoc_trkPF.trackType = 'inner'
NEWrecoMuonVMuAssoc_trkPF.selection = "isTrackerMuon & isPFMuon"

#standalone
NEWmuonAssociatorByHitsNoSimHitsHelperStandalone = muonAssociatorByHitsNoSimHitsHelper.clone()
NEWmuonAssociatorByHitsNoSimHitsHelperStandalone.UseTracker = False
NEWmuonAssociatorByHitsNoSimHitsHelperStandalone.UseMuon  = True
NEWrecoMuonVMuAssoc_sta = NewRecoMuonValidator.clone()
NEWrecoMuonVMuAssoc_sta.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Sta'
NEWrecoMuonVMuAssoc_sta.muAssocLabel = 'NEWmuonAssociatorByHitsNoSimHitsHelperStandalone'
NEWrecoMuonVMuAssoc_sta.trackType = 'outer'
NEWrecoMuonVMuAssoc_sta.selection = "isStandAloneMuon"

#seed of StandAlone
NEWmuonAssociatorByHitsNoSimHitsHelperSeedStandalone = muonAssociatorByHitsNoSimHitsHelper.clone()
NEWmuonAssociatorByHitsNoSimHitsHelperSeedStandalone.UseTracker = False
NEWmuonAssociatorByHitsNoSimHitsHelperSeedStandalone.UseMuon  = True
NEWrecoMuonVMuAssoc_seedSta = NewRecoMuonValidator.clone()
NEWrecoMuonVMuAssoc_seedSta.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_SeedSta'
NEWrecoMuonVMuAssoc_seedSta.muAssocLabel = 'NEWmuonAssociatorByHitsNoSimHitsHelperStandalone'
NEWrecoMuonVMuAssoc_seedSta.trackType = 'outer'
NEWrecoMuonVMuAssoc_seedSta.selection = ""

#standalone and PF
NEWmuonAssociatorByHitsNoSimHitsHelperStandalonePF = muonAssociatorByHitsNoSimHitsHelper.clone()
NEWmuonAssociatorByHitsNoSimHitsHelperStandalonePF.UseTracker = False
NEWmuonAssociatorByHitsNoSimHitsHelperStandalonePF.UseMuon  = True
NEWrecoMuonVMuAssoc_staPF = NewRecoMuonValidator.clone()
NEWrecoMuonVMuAssoc_staPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_StaPF'
NEWrecoMuonVMuAssoc_staPF.usePFMuon = True
NEWrecoMuonVMuAssoc_staPF.muAssocLabel = 'NEWmuonAssociatorByHitsNoSimHitsHelperStandalonePF'
NEWrecoMuonVMuAssoc_staPF.trackType = 'outer'
NEWrecoMuonVMuAssoc_staPF.selection = "isStandAloneMuon & isPFMuon"

#global
NEWmuonAssociatorByHitsNoSimHitsHelperGlobal = muonAssociatorByHitsNoSimHitsHelper.clone()
NEWmuonAssociatorByHitsNoSimHitsHelperGlobal.UseTracker = True
NEWmuonAssociatorByHitsNoSimHitsHelperGlobal.UseMuon  = True
NEWrecoMuonVMuAssoc_glb = NewRecoMuonValidator.clone()
NEWrecoMuonVMuAssoc_glb.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Glb'
NEWrecoMuonVMuAssoc_glb.muAssocLabel = 'NEWmuonAssociatorByHitsNoSimHitsHelperGlobal'
NEWrecoMuonVMuAssoc_glb.trackType = 'global'
NEWrecoMuonVMuAssoc_glb.selection = "isGlobalMuon"

#global and PF
NEWmuonAssociatorByHitsNoSimHitsHelperGlobalPF = muonAssociatorByHitsNoSimHitsHelper.clone()
NEWmuonAssociatorByHitsNoSimHitsHelperGlobalPF.UseTracker = True
NEWmuonAssociatorByHitsNoSimHitsHelperGlobalPF.UseMuon  = True
NEWrecoMuonVMuAssoc_glbPF = NewRecoMuonValidator.clone()
NEWrecoMuonVMuAssoc_glbPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_GlbPF'
NEWrecoMuonVMuAssoc_glbPF.usePFMuon = True
NEWrecoMuonVMuAssoc_glbPF.muAssocLabel = 'NEWmuonAssociatorByHitsNoSimHitsHelperGlobalPF'
NEWrecoMuonVMuAssoc_glbPF.trackType = 'global'
NEWrecoMuonVMuAssoc_glbPF.selection = "isGlobalMuon & isPFMuon"

#tight
NEWmuonAssociatorByHitsNoSimHitsHelperTight = muonAssociatorByHitsNoSimHitsHelper.clone()
NEWmuonAssociatorByHitsNoSimHitsHelperTight.UseTracker = True
NEWmuonAssociatorByHitsNoSimHitsHelperTight.UseMuon  = True
NEWrecoMuonVMuAssoc_tgt = NewRecoMuonValidator.clone()
NEWrecoMuonVMuAssoc_tgt.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Tgt'
NEWrecoMuonVMuAssoc_tgt.muAssocLabel = 'NEWmuonAssociatorByHitsNoSimHitsHelperTight'
NEWrecoMuonVMuAssoc_tgt.trackType = 'global'
NEWrecoMuonVMuAssoc_tgt.selection = 'isGlobalMuon'
NEWrecoMuonVMuAssoc_tgt.wantTightMuon = True
NEWrecoMuonVMuAssoc_tgt.beamSpot = 'offlineBeamSpot'
NEWrecoMuonVMuAssoc_tgt.primaryVertex = 'offlinePrimaryVertices'

##########################################################################
# Muon validation sequence using RecoMuonValidator
#

NEWmuonValidationRMV_seq = cms.Sequence(
    NEWmuonAssociatorByHitsNoSimHitsHelperTrk +NEWrecoMuonVMuAssoc_trk
    +NEWmuonAssociatorByHitsNoSimHitsHelperStandalone +NEWrecoMuonVMuAssoc_sta
    +NEWmuonAssociatorByHitsNoSimHitsHelperGlobal +NEWrecoMuonVMuAssoc_glb
    +NEWmuonAssociatorByHitsNoSimHitsHelperTight +NEWrecoMuonVMuAssoc_tgt
    # 
    #    +NEWmuonAssociatorByHitsNoSimHitsHelperTrkPF +NEWrecoMuonVMuAssoc_trkPF
    #    +NEWmuonAssociatorByHitsNoSimHitsHelperStandalonePF +NEWrecoMuonVMuAssoc_staPF
    #    +NEWmuonAssociatorByHitsNoSimHitsHelperGlobalPF +NEWrecoMuonVMuAssoc_glbPF
    )

