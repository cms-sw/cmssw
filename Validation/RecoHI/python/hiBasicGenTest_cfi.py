import FWCore.ParameterSet.Config as cms

hiBasicGenTest = DQMStep1Module('HiBasicGenTest',
                                generatorLabel = cms.InputTag('generatorSmeared'),
                                outputFile = cms.string('')
)
