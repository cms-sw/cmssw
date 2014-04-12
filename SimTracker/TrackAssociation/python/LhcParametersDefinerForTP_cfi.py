import FWCore.ParameterSet.Config as cms

LhcParametersDefinerForTP = cms.ESProducer("ParametersDefinerForTPESProducer",
   ComponentName = cms.string('LhcParametersDefinerForTP'),
   beamSpot      = cms.untracked.InputTag('offlineBeamSpot')                                        
)
