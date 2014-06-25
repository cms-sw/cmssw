import FWCore.ParameterSet.Config as cms

hgcalSimHitClientEE = cms.EDAnalyzer("HGCalSimHitsClient", 
                                     outputFile = cms.untracked.string(''),
                                     DetectorName = cms.string("HGCalEESensitive"),
)
