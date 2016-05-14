import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.NewMuonValidation_cff import *

# pt-selection of reco tracks
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
cutsRecoTrkMuons = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTrkMuons.src = "hiGeneralTracks"
cutsRecoTrkMuons.quality = []
cutsRecoTrkMuons.ptMin = 0.0

# pt-selection of tracking particles
import PhysicsTools.RecoAlgos.trackingParticleSelector_cfi
cutsTpMuons = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone()
cutsTpMuons.ptMin = 0.0

#----------------------------------------

# MuonAssociation labels; hit-by-hit matching only,MuonAssociator
#
import SimMuon.MCTruth.NewMuonAssociatorByHits_cfi
MABH = SimMuon.MCTruth.NewMuonAssociatorByHits_cfi.NewMuonAssociatorByHits.clone()
# DEFAULTS ###################################
#    EfficiencyCut_track = cms.double(0.),
#    PurityCut_track = cms.double(0.),
#    EfficiencyCut_muon = cms.double(0.),
#    PurityCut_muon = cms.double(0.),
#    includeZeroHitMuons = cms.bool(True),
#    acceptOneStubMatchings = cms.bool(False),
##############################################
MABH.tpTag = 'cutsTpMuons'
MABH.acceptOneStubMatchings = cms.bool(True) # OLD setting: should be kept at default ("False")
#MABH.EfficiencyCut_track = 0.5
MABH.PurityCut_track = 0.75
MABH.PurityCut_muon = 0.75
#MABH.includeZeroHitMuons = False
################################################

# sim to tracker tracks, 
NEWtpToTkMuonAssociation = MABH.clone()
NEWtpToTkMuonAssociation.tracksTag = 'cutsRecoTrkMuons'
NEWtpToTkMuonAssociation.UseTracker = True
NEWtpToTkMuonAssociation.UseMuon = False

# sim to sta, and sta:updatedAtVtx
NEWtpToStaMuonAssociation = MABH.clone()
NEWtpToStaMuonAssociation.tracksTag = 'standAloneMuons'
NEWtpToStaMuonAssociation.UseTracker = False
NEWtpToStaMuonAssociation.UseMuon = True

NEWtpToStaUpdMuonAssociation = MABH.clone()
NEWtpToStaUpdMuonAssociation.tracksTag = 'standAloneMuons:UpdatedAtVtx'
NEWtpToStaUpdMuonAssociation.UseTracker = False
NEWtpToStaUpdMuonAssociation.UseMuon = True

# sim to glb track 
NEWtpToGlbMuonAssociation = MABH.clone()
NEWtpToGlbMuonAssociation.tracksTag = 'globalMuons'
NEWtpToGlbMuonAssociation.UseTracker = True
NEWtpToGlbMuonAssociation.UseMuon = True


# Muon association sequences
# (some are commented out until timing is addressed)
NEWhiMuonAssociation_seq = cms.Sequence(
    NEWtpToTkMuonAssociation+
    NEWtpToStaMuonAssociation+
    NEWtpToStaUpdMuonAssociation+
    NEWtpToGlbMuonAssociation
    )

#----------------------------------------

# RecoMuonValidators labels
NEWtrkMuonTrackVTrackAssoc.associatormap  = 'NEWtpToTkMuonAssociation'
NEWtrkMuonTrackVTrackAssoc.associators    = 'MuonAssociationByHits'
NEWtrkMuonTrackVTrackAssoc.label          = ['cutsRecoTrkMuons']
NEWtrkMuonTrackVTrackAssoc.label_tp_effic = 'cutsTpMuons'
NEWtrkMuonTrackVTrackAssoc.label_tp_fake  = 'cutsTpMuons'

NEWglbMuonTrackVMuonAssoc.label           = ['globalMuons']
NEWglbMuonTrackVMuonAssoc.label_tp_effic  = 'cutsTpMuons'
NEWglbMuonTrackVMuonAssoc.label_tp_fake   = 'cutsTpMuons'

NEWstaMuonTrackVMuonAssoc.label_tp_effic  = 'cutsTpMuons'
NEWstaMuonTrackVMuonAssoc.label_tp_fake  = 'cutsTpMuons'

NEWstaUpdMuonTrackVMuonAssoc.label_tp_effic  = 'cutsTpMuons'
NEWstaUpdMuonTrackVMuonAssoc.label_tp_fake  = 'cutsTpMuons'

#change pt max of track validator
NEWtrkMuonTrackVTrackAssoc.maxPt = cms.double(100)
NEWglbMuonTrackVMuonAssoc.maxPt = cms.double(100)
NEWstaMuonTrackVMuonAssoc.maxPt = cms.double(100)
NEWstaUpdMuonTrackVMuonAssoc.maxPt = cms.double(100)

# Muon validation sequences
NEWhiMuonValidation_seq = cms.Sequence(
    NEWtrkMuonTrackVTrackAssoc+
    NEWstaMuonTrackVMuonAssoc+
    NEWstaUpdMuonTrackVMuonAssoc+
    NEWglbMuonTrackVMuonAssoc
    )

#----------------------------------------

# HI muon prevalidation
NEWhiRecoMuonPrevalidation = cms.Sequence(
    cutsRecoTrkMuons
  * cutsTpMuons
  * NEWhiMuonAssociation_seq
)

# HI muon validation sequence
NEWhiRecoMuonValidation = cms.Sequence( NEWhiMuonValidation_seq )    
