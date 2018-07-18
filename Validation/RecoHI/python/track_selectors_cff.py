import FWCore.ParameterSet.Config as cms

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

