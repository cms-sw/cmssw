import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.muonValidation_cff import *

# pt-selection of reco tracks
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
cutsRecoTrkMuons = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTrkMuons.src = "hiSelectedTracks"
cutsRecoTrkMuons.quality = []
cutsRecoTrkMuons.ptMin = 2.0

# pt-selection of tracking particles
import PhysicsTools.RecoAlgos.trackingParticleSelector_cfi
cutsTpMuons = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone()
cutsTpMuons.ptMin = 2.0

#----------------------------------------

# MuonAssociation labels; hit-by-hit matching only,MuonAssociator

# sim to tracker tracks, 
tpToTkMuonAssociation.tracksTag = 'cutsRecoTrkMuons'
tpToTkMuonAssociation.tpTag     = 'cutsTpMuons'

# sim to sta, and sta:updatedAtVtx
tpToStaMuonAssociation.tpTag    = 'cutsTpMuons'
tpToStaUpdMuonAssociation.tpTag = 'cutsTpMuons'

# sim to glb track 
tpToGlbMuonAssociation.tpTag    = 'cutsTpMuons'
tpToGlbMuonAssociation.tracksTag = 'globalMuons'

# MuonAssociation cuts for heavy ion events
tpToTkMuonAssociation.PurityCut_track = 0.75
tpToStaMuonAssociation.UseMuon = True
tpToStaMuonAssociation.PurityCut_muon = 0.75
tpToStaUpdMuonAssociation.UseMuon = True
tpToStaUpdMuonAssociation.PurityCut_muon = 0.75
tpToGlbMuonAssociation.UseTracker = True
tpToGlbMuonAssociation.PurityCut_track = 0.75
tpToGlbMuonAssociation.UseMuon = True
tpToGlbMuonAssociation.PurityCut_muon = 0.75

# Muon association sequences
# (some are commented out until timing is addressed)
hiMuonAssociation_seq = cms.Sequence(
    tpToTkMuonAssociation+
    tpToStaMuonAssociation+
    tpToStaUpdMuonAssociation+
    tpToGlbMuonAssociation
    )

#----------------------------------------

# RecoMuonValidators labels
trkMuonTrackVTrackAssoc.associatormap  = 'tpToTkMuonAssociation'
trkMuonTrackVTrackAssoc.label          = ['cutsRecoTrkMuons']
trkMuonTrackVTrackAssoc.label_tp_effic = 'cutsTpMuons'
trkMuonTrackVTrackAssoc.label_tp_fake  = 'cutsTpMuons'

glbMuonTrackVMuonAssoc.label           = ['globalMuons']
glbMuonTrackVMuonAssoc.label_tp_effic  = 'cutsTpMuons'
glbMuonTrackVMuonAssoc.label_tp_fake   = 'cutsTpMuons'

staMuonTrackVMuonAssoc.label_tp_effic  = 'cutsTpMuons'
staMuonTrackVMuonAssoc.label_tp_fake  = 'cutsTpMuons'

staUpdMuonTrackVMuonAssoc.label_tp_effic  = 'cutsTpMuons'
staUpdMuonTrackVMuonAssoc.label_tp_fake  = 'cutsTpMuons'

#change pt max of track validator
trkMuonTrackVTrackAssoc.maxpT = cms.double(200)
glbMuonTrackVMuonAssoc.maxpT = cms.double(200)
staMuonTrackVMuonAssoc.maxpT = cms.double(200)
staUpdMuonTrackVMuonAssoc.maxpT = cms.double(200)

# Muon validation sequences
hiMuonValidation_seq = cms.Sequence(
    trkMuonTrackVTrackAssoc+
    staMuonTrackVMuonAssoc+
    staUpdMuonTrackVMuonAssoc+
    glbMuonTrackVMuonAssoc
    )

#----------------------------------------

# HI muon validation sequence
hiRecoMuonValidation = cms.Sequence(cutsRecoTrkMuons *
                                    cutsTpMuons *
                                    hiMuonAssociation_seq *
                                    hiMuonValidation_seq)    
