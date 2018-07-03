import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hiBasicGenTest = DQMEDAnalyzer('HiBasicGenTest',
                                generatorLabel = cms.InputTag('generatorSmeared'),
                                outputFile = cms.string('')
)
