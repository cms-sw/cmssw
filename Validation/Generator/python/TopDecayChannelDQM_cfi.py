import FWCore.ParameterSet.Config as cms

topDecayChannelDQM = cms.EDAnalyzer('TopDecayChannelDQM',
  ## input collection; this maybe any collection of type
  ## reco::GenParticles.
  src = cms.InputTag('genParticles'),
  ## number of events for which a full partile listing
  ## will be printed into the logfile. 0 will switch
  ## this kind of logging off.
  logEvents = cms.uint32(10)
)
