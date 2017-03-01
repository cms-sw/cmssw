import FWCore.ParameterSet.Config as cms

hiBasicGenTest = cms.EDAnalyzer("HiBasicGenTest",
                                generatorLabel = cms.InputTag('generatorSmeared'),
                                outputFile = cms.string('')
)
