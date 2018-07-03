from SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi import * 

muonSimClassifier = cms.EDProducer("MuonSimClassifier",
    muons = cms.InputTag("muons"),
    trackType = cms.string("glb_or_trk"),  # 'inner','outer','global','segments','glb_or_trk'
    trackingParticles = cms.InputTag("mix","MergedTrackTruth"), # default TrackingParticle collection (should exist in the Event)      
    associatorLabel   = cms.InputTag("muonAssociatorByHitsNoSimHitsHelper"),
    decayRho  = cms.double(200), # to classify differently decay muons included in ppMuX
    decayAbsZ = cms.double(400), # and decay muons that could not be in ppMuX
    linkToGenParticles = cms.bool(True),          # produce also a collection of GenParticles for secondary muons
    genParticles = cms.InputTag("genParticles"),  # and associations to primary and secondaries
)

muonSimClassificationByHitsTask = cms.Task(
    muonAssociatorByHitsNoSimHitsHelper,muonSimClassifier
)
