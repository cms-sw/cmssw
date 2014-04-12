import FWCore.ParameterSet.Config as cms

crackseedfinder = cms.EDProducer("CRackSeedGenerator",
                               matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
                               rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
                               stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
                               ptMin = cms.double(200000.0), #no ms
                               SeedPt = cms.double(1.0),
                               originRadius = cms.double(150.0),
                               originHalfLength = cms.double(90.0),
                               originZPosition = cms.double(0.0),
                               TTRHBuilder = cms.string('WithTrackAngle'),
                               GeometricStructure = cms.untracked.string('CRACK')
                               )
