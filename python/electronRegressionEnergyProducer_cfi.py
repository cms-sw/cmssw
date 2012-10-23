import FWCore.ParameterSet.Config as cms

eleRegressionEnergy = cms.EDProducer("RegressionEnergyPatElectronProducer",
                                   debug = cms.untracked.bool(False),
                                   inputElectronsTag = cms.InputTag('cleanPatElectrons'),
                                     # inputCollectionType (0: GsfElectron, 1 :PATElectron)
                                   inputCollectionType = cms.uint32(1),
                                     # useRecHitCollections ( should be set to true if GsfElectrons or if the RecHits have not been embedded into the PATElectron
                                   useRecHitCollections = cms.bool(False),
                                     # produce ValueMaps. Should be true for GsfElectrons otherwise this producer doest nothing
                                   produceValueMaps = cms.bool(False),
                                   regressionInputFile = cms.string("EGamma/EGammaAnalysisTools/data/eleEnergyRegWeights_V1.root"),
                                   energyRegressionType = cms.uint32(1),
                                   rhoCollection = cms.InputTag('kt6PFJets:rho:RECO'),
                                   vertexCollection = cms.InputTag('offlinePrimaryVertices')  
                                   nameEnergyReg = cms.string("eneRegForGsfEle"),
                                   nameEnergyErrorReg = cms.string("eneErrorRegForGsfEle"),
                                   recHitCollectionEB = cms.InputTag('reducedEcalRecHitsEB'),
                                   recHitCollectionEE = cms.InputTag('reducedEcalRecHitsEE')
                                   )
