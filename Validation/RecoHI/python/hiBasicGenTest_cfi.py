import FWCore.ParameterSet.Config as cms

hiBasicGenTest = cms.EDAnalyzer("HiBasicGenTest",
                                generatorLabel = cms.InputTag('generator'),
                                outputFile = cms.string('')
)
