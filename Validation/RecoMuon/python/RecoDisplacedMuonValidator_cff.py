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
muonAssociatorByHitsNoSimHitsHelperTrk = muonAssociatorByHitsNoSimHitsHelper.clone(
    UseTracker = True,
    UseMuon  = False
)
recoDisplacedMuonVMuAssoc_trk = recoDisplacedMuonValidator.clone(
    subDir = 'Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Trk',
    muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperTrk',
    trackType = 'inner',
    selection = "isTrackerMuon",
    simLabel = ("TPmu"),
    tpRefVector = True
)
recoDisplacedMuonVMuAssoc_trk.tpSelector.src = ("TPmu")
#standalone
muonAssociatorByHitsNoSimHitsHelperStandalone = muonAssociatorByHitsNoSimHitsHelper.clone(
    UseTracker = False,
    UseMuon  = True
)
recoDisplacedMuonVMuAssoc_sta = recoDisplacedMuonValidator.clone(
    subDir = 'Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Sta',
    muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperStandalone',
    trackType = 'outer',
    selection = "isStandAloneMuon",
    simLabel = ("TPmu"),
    tpRefVector = True

)
recoDisplacedMuonVMuAssoc_sta.tpSelector.src = ("TPmu")
#global
muonAssociatorByHitsNoSimHitsHelperGlobal = muonAssociatorByHitsNoSimHitsHelper.clone(
    UseTracker = True,
    UseMuon  = True
)
recoDisplacedMuonVMuAssoc_glb = recoDisplacedMuonValidator.clone(
    subDir = 'Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Glb',
    muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperGlobal',
    trackType = 'global',
    selection = "isGlobalMuon",
    simLabel = ("TPmu"),
    tpRefVector = True,
)
recoDisplacedMuonVMuAssoc_glb.tpSelector.src = ("TPmu")
#tight
muonAssociatorByHitsNoSimHitsHelperTight = muonAssociatorByHitsNoSimHitsHelper.clone(
    UseTracker = True,
    UseMuon  = True
)
recoDisplacedMuonVMuAssoc_tgt = recoDisplacedMuonValidator.clone(
    subDir = 'Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Tgt',
    muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperTight',
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
    muonAssociatorByHitsNoSimHitsHelperTrk+recoDisplacedMuonVMuAssoc_trk
    +muonAssociatorByHitsNoSimHitsHelperStandalone+recoDisplacedMuonVMuAssoc_sta
    +muonAssociatorByHitsNoSimHitsHelperGlobal+recoDisplacedMuonVMuAssoc_glb
    +muonAssociatorByHitsNoSimHitsHelperTight+recoDisplacedMuonVMuAssoc_tgt
    )
