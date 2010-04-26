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

# MuonAssociation labels
tpToTkMuonAssociation.tracksTag = 'cutsRecoTrkMuons'
tpToTkMuonAssociation.tpTag = 'cutsTpMuons'
tpToStaMuonAssociation.tpTag = 'cutsTpMuons'
tpToStaUpdMuonAssociation.tpTag = 'cutsTpMuons'
tpToGlbMuonAssociation.tpTag = 'cutsTpMuons'

tpToTkmuTrackAssociation.label_tr = 'cutsRecoTrkMuons'
tpToTkmuTrackAssociation.label_tp = 'cutsTpMuons'
tpToStaTrackAssociation.label_tp = 'cutsTpMuons'
tpToStaUpdTrackAssociation.label_tp = 'cutsTpMuons'
tpToGlbTrackAssociation.label_tp = 'cutsTpMuons'


# Muon association sequences
# (some are commented out until timing is addressed)
hiMuonAssociation_seq = cms.Sequence(
    tpToTkMuonAssociation+
    tpToStaMuonAssociation+
    tpToStaUpdMuonAssociation+
    tpToGlbMuonAssociation+
    tpToTkmuTrackAssociation+
    tpToStaTrackAssociation+
    tpToStaUpdTrackAssociation+
    tpToGlbTrackAssociation
    )

#----------------------------------------

# RecoMuonValidators labels
trkMuonTrackVTrackAssoc.label = ['cutsRecoTrkMuons']
recoMuonVMuAssoc.trkMuLabel = 'cutsRecoTrkMuons'
recoMuonVTrackAssoc.trkMuLabel = 'cutsRecoTrkMuons'

# Muon validation sequences
hiMuonValidation_seq = cms.Sequence(
    trkMuonTrackVTrackAssoc+
    staMuonTrackVTrackAssoc+
    staUpdMuonTrackVTrackAssoc+
    glbMuonTrackVTrackAssoc+
    staMuonTrackVMuonAssoc+
    staUpdMuonTrackVMuonAssoc+
    glbMuonTrackVMuonAssoc+
    recoMuonVMuAssoc+
    recoMuonVTrackAssoc
    )

#----------------------------------------

# HI muon validation sequence
hiRecoMuonValidation = cms.Sequence(cutsRecoTrkMuons *
                                    cutsTpMuons *
                                    hiMuonAssociation_seq *
                                    hiMuonValidation_seq)    
