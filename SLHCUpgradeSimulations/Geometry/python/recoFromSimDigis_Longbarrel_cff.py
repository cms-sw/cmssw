import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import *
siPixelClusters.src = 'simSiPixelDigis'
siPixelClusters.MissCalibrate = False

from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
siStripZeroSuppression.RawDigiProducersList = cms.VInputTag( cms.InputTag('simSiStripDigis','VirginRaw'),
                                                                     cms.InputTag('simSiStripDigis','ProcessedRaw'),
                                                                     cms.InputTag('simSiStripDigis','ScopeMode'))
#siStripZeroSuppression.RawDigiProducersList[0].RawDigiProducer = ''
#siStripZeroSuppression.RawDigiProducersList[1].RawDigiProducer = ''
#siStripZeroSuppression.RawDigiProducersList[2].RawDigiProducer = ''

from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
siStripClusters.DigiProducersList = cms.VInputTag(cms.InputTag('simSiStripDigis','ZeroSuppressed'),
                                                          cms.InputTag('siStripZeroSuppression','VirginRaw'),
                                                          cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
                                                          cms.InputTag('siStripZeroSuppression','ScopeMode'))
#siStripClusters.DigiProducersList[0].DigiProducer= ''

from SimTracker.SiStripDigitizer.SiStripDigi_cfi import *
simSiStripDigis.ROUList = cms.vstring("g4SimHitsTrackerHitsPixelBarrelLowTof")
