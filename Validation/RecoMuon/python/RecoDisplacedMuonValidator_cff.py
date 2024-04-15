import FWCore.ParameterSet.Config as cms

#####################################################################################
# Configurations for RecoDisplacedMuonValidator
#

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoDisplacedMuonValidator_cfi import *
#
from SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi import *
from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters

#tracker
muondispAssociatorByHitsNoSimHitsHelperTrk = muonAssociatorByHitsNoSimHitsHelper.clone(
    UseTracker = True,
    UseMuon  = False
)
recoDisplacedMuonVMuAssoc_trk = recoDisplacedMuonValidator.clone(
    subDir = 'Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Trk',
    muAssocLabel = 'muondispAssociatorByHitsNoSimHitsHelperTrk',
    trackType = 'inner',
    selection = "isTrackerMuon",
    simLabel = ("TPmu"),
    tpRefVector = True
)
recoDisplacedMuonVMuAssoc_trk.tpSelector.src = ("TPmu")
#standalone
muondispAssociatorByHitsNoSimHitsHelperStandalone = muonAssociatorByHitsNoSimHitsHelper.clone(
    UseTracker = False,
    UseMuon  = True
)
recoDisplacedMuonVMuAssoc_sta = recoDisplacedMuonValidator.clone(
    subDir = 'Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Sta',
    muAssocLabel = 'muondispAssociatorByHitsNoSimHitsHelperStandalone',
    trackType = 'outer',
    selection = "isStandAloneMuon",
    simLabel = ("TPmu"),
    tpRefVector = True,
    nBinDxy = cms.untracked.uint32(100),
    minDxy = cms.untracked.double(-350),
    maxDxy = cms.untracked.double(350),
    nBinDz = cms.untracked.uint32(100),
    minDz = cms.untracked.double(-350),
    maxDz = cms.untracked.double(350)
)
recoDisplacedMuonVMuAssoc_sta.tpSelector.src = ("TPmu")
#global
muondispAssociatorByHitsNoSimHitsHelperGlobal = muonAssociatorByHitsNoSimHitsHelper.clone(
    UseTracker = True,
    UseMuon  = True
)
recoDisplacedMuonVMuAssoc_glb = recoDisplacedMuonValidator.clone(
    subDir = 'Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Glb',
    muAssocLabel = 'muondispAssociatorByHitsNoSimHitsHelperGlobal',
    trackType = 'global',
    selection = "isGlobalMuon",
    simLabel = ("TPmu"),
    tpRefVector = True,
)
recoDisplacedMuonVMuAssoc_glb.tpSelector.src = ("TPmu")
#tight
muondispAssociatorByHitsNoSimHitsHelperTight = muonAssociatorByHitsNoSimHitsHelper.clone(
    UseTracker = True,
    UseMuon  = True
)
recoDisplacedMuonVMuAssoc_tgt = recoDisplacedMuonValidator.clone(
    subDir = 'Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Tgt',
    muAssocLabel = 'muondispAssociatorByHitsNoSimHitsHelperTight',
    trackType = 'global',
    selection = 'isGlobalMuon',
    wantTightMuon = True,
    beamSpot = 'offlineBeamSpot',
    primaryVertex = 'offlinePrimaryVertices',
    simLabel = ("TPmu"),
    tpRefVector = True,
)
recoDisplacedMuonVMuAssoc_tgt.tpSelector.src = ("TPmu")
##########################################################################
# Muon validation sequence using RecoDisplacedMuonValidator
#

muonValidationRDMV_seq = cms.Sequence(
    muondispAssociatorByHitsNoSimHitsHelperTrk+recoDisplacedMuonVMuAssoc_trk
    +muondispAssociatorByHitsNoSimHitsHelperStandalone+recoDisplacedMuonVMuAssoc_sta
    +muondispAssociatorByHitsNoSimHitsHelperGlobal+recoDisplacedMuonVMuAssoc_glb
    +muondispAssociatorByHitsNoSimHitsHelperTight+recoDisplacedMuonVMuAssoc_tgt
    )
