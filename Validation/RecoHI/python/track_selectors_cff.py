import FWCore.ParameterSet.Config as cms

# pt-selection of reco tracks
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
NEWcutsRecoTrkMuons = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
NEWcutsRecoTrkMuons.src = "hiGeneralTracks"
NEWcutsRecoTrkMuons.quality = []
NEWcutsRecoTrkMuons.ptMin = 0.0

# pt-selection of tracking particles
import PhysicsTools.RecoAlgos.trackingParticleSelector_cfi
NEWcutsTpMuons = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone()
NEWcutsTpMuons.ptMin = 0.0

