import FWCore.ParameterSet.Config as cms

MuonImput = cms.InputTag("selectedMuonsForEmbedding","","")

cleanedsiPixelDigis = cms.EDProducer('PixelDigiCleaner',
    MuonCollection = MuonImput,
    oldCollection = cms.InputTag("siPixelDigis","","")
)



cleanedsiPixelClusters = cms.EDProducer('PixelCleaner',
    MuonCollection = MuonImput,
    oldCollection = cms.InputTag("siPixelClusters","","")
)

cleanedsiPixelClustersPreSplitting = cms.EDProducer('PixelCleaner',
    MuonCollection = MuonImput,
    oldCollection = cms.InputTag("siPixelClustersPreSplitting","","")
)

cleanedsiStripClusters = cms.EDProducer('StripCleaner',
    MuonCollection = MuonImput,
    oldCollection = cms.InputTag("siStripClusters","",""),
)

from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

cleanedecalRecHit = cms.EDProducer("EcalRecHitCleaner",
    MuonCollection = MuonImput,
    TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,
    oldCollection = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB",""),
    cms.InputTag("ecalRecHit","EcalRecHitsEE",""))
)

cleanedecalPreshowerRecHit = cms.EDProducer("EcalRecHitCleaner",
    MuonCollection = MuonImput,
    TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,
    oldCollection = cms.VInputTag(cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES",""))
)
cleanedecalPreshowerRecHit.TrackAssociatorParameters.usePreshower = cms.bool(True) 

cleanedhbhereco = cms.EDProducer("HBHERecHitCleaner",
    MuonCollection = MuonImput,
    TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,
    oldCollection = cms.VInputTag(cms.InputTag("hbhereco","",""))
)

cleanedhoreco = cms.EDProducer("HORecHitCleaner",
    MuonCollection = MuonImput,
    TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,
    oldCollection = cms.VInputTag(cms.InputTag("horeco","",""))
)


cleanedzdcreco = cms.EDProducer('ZDCRecHitCleaner',
    MuonCollection = MuonImput,
    TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,
    oldCollection  = cms.VInputTag(cms.InputTag("zdcreco","",""))
)

cleanedhbheprereco = cms.EDProducer('HBHERecHitCleaner',
    MuonCollection = MuonImput,
    TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,
    oldCollection  = cms.VInputTag(cms.InputTag("hbheprereco","",""))
)

cleanedhfreco = cms.EDProducer('HFRecHitCleaner',
    MuonCollection = MuonImput,
    TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,
    oldCollection  = cms.VInputTag(cms.InputTag("hfreco","",""))
)

cleanedcastorreco = cms.EDProducer('CastorRecHitCleaner',
    MuonCollection = MuonImput,
    TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,
    oldCollection  = cms.VInputTag(cms.InputTag("castorreco","",""))
)






cleaneddt1DRecHits = cms.EDProducer('DTCleaner',
    MuonCollection = MuonImput,
    oldCollection = cms.InputTag("dt1DRecHits","",""),
)

cleanedcsc2DRecHits = cms.EDProducer('CSCCleaner',
    MuonCollection = MuonImput,
    oldCollection  = cms.InputTag("csc2DRecHits","",""),
)

cleanedrpcRecHits = cms.EDProducer('RPCleaner',
    MuonCollection = MuonImput,
    oldCollection  = cms.InputTag("rpcRecHits","",""),
)

cleaneddt1DCosmicRecHits = cms.EDProducer('DTCleaner',
    MuonCollection = MuonImput,
    oldCollection  = cms.InputTag("dt1DCosmicRecHits","",""),
)






## Nothing to clean for this collections, but keep a copy of them ;)






makeCleaningProcedure = cms.Sequence(
  #  cleanedsiPixelDigis
     cleanedsiPixelClusters
  #  + cleanedsiPixelClustersPreSplitting
    + cleanedsiStripClusters
    + cleanedecalRecHit
    + cleanedecalPreshowerRecHit
    + cleanedhbhereco
    + cleanedhoreco
    + cleaneddt1DRecHits
    + cleanedcsc2DRecHits
    + cleanedrpcRecHits
    + cleanedzdcreco
    + cleanedhbheprereco
    + cleaneddt1DCosmicRecHits
    + cleanedhfreco
    + cleanedcastorreco
)
