import FWCore.ParameterSet.Config as cms

eleRegressionEnergy = cms.EDProducer("RegressionEnergyPatElectronProducer",
                                   printDebug = cms.untracked.bool(False),
                                   inputPatElectronsTag = cms.InputTag('selectedPatElectrons'),
                                   regressionInputFile = cms.string("EGamma/EGammaAnalysisTools/data/eleEnergyRegWeights_V1.root"),
                                   energyRegressionType = cms.uint32(1),
                                   debug = cms.bool(False)  
#                                   nameEnergyReg = cms.string("eneRegForGsfEle"),
#                                   nameEnergyErrorReg = cms.string("eneErrorRegForGsfEle"),
#                                   recHitCollectionEB = cms.InputTag('reducedEcalRecHitsEB'),
#                                   recHitCollectionEE = cms.InputTag('reducedEcalRecHitsEE')

                                   )
