import FWCore.ParameterSet.Config as cms

trackingParticleMuon = cms.EDProducer("TrackingParticleRefMuonProducer",
   mightGet = cms.optional.untracked.vstring,
   src = cms.InputTag("mix","MergedTrackTruth"),
   skim = cms.string('mu'),
#   skim = cms.string('track'),
#   skim = cms.string('pf'),
   ptmin = cms.double(0.5),
   pmin = cms.double(2.5)
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(trackingParticleMuon, src = "mixData:MergedTrackTruth")
