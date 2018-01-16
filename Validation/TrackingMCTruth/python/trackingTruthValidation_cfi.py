import FWCore.ParameterSet.Config as cms

trackingTruthValid = DQMStep1Module('TrackingTruthValid',
    #to run on original collection
    #   InputTag src = trackingtruthprod:
    #   string outputFile = "trackingtruthhisto.root" 
    #to run on merged collection (default)
    src = cms.InputTag("mix","MergedTrackTruth"),
    runStandalone = cms.bool(False),
    outputFile = cms.string('')
)


