import FWCore.ParameterSet.Config as cms

L1CaloTriggerSetupSource = cms.ESSource("EmptyESSource",
                                        recordName = cms.string('L1CaloTriggerSetupRcd'),
                                        firstValid = cms.vuint32(1),
                                        iovIsRunNotTime = cms.bool(True)
                                        )

L1CaloTriggerSetup = cms.ESProducer("L1CaloTriggerSetupProducer",
                                    InputXMLFile = cms.FileInPath('SLHCUpgradeSimulations/L1CaloTrigger/data/setup.xml')
                                    )

L1CaloTowerProducer = cms.EDProducer("L1CaloTowerProducer",
    ECALDigis = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    HCALDigis = cms.InputTag("simHcalTriggerPrimitiveDigis")
)

L1CaloRegionProducer = cms.EDProducer("L1CaloRegionProducer",
                                      src = cms.InputTag("L1CaloTowerProducer")
)                                      

L1CaloClusterProducer = cms.EDProducer("L1CaloClusterProducer",
    src = cms.InputTag("L1CaloTowerProducer")
)

L1CaloClusterFilter = cms.EDProducer("L1CaloClusterFilter",
    src = cms.InputTag("L1CaloClusterProducer")
)

L1CaloClusterIsolator = cms.EDProducer("L1CaloClusterIsolator",
    src = cms.InputTag("L1CaloClusterFilter")
)

L1CaloJetProducer = cms.EDProducer("L1CaloJetProducer",
    src = cms.InputTag("L1CaloRegionProducer")
)

L1CaloJetFilter = cms.EDProducer("L1CaloJetFilter",
    src = cms.InputTag("L1CaloJetProducer")
)

L1CaloJetExpander = cms.EDProducer("L1CaloJetExpander",
    src = cms.InputTag("L1CaloJetFilter")
)


rawSLHCL1ExtraParticles = cms.EDProducer("L1ExtraTranslator",
                                  Clusters = cms.InputTag("L1CaloClusterIsolator"),
                                  Jets = cms.InputTag("L1CaloJetExpander"),
                                  NParticles = cms.int32(8),
                                  NJets      = cms.int32(12)
                              
)

SLHCL1ExtraParticles = cms.EDProducer("L1ExtraCalibrator",
                                      eGamma = cms.InputTag("rawSLHCL1ExtraParticles","EGamma"),
                                      isoEGamma = cms.InputTag("rawSLHCL1ExtraParticles","IsoEGamma"),
                                      taus = cms.InputTag("rawSLHCL1ExtraParticles","Taus"),
                                      isoTaus = cms.InputTag("rawSLHCL1ExtraParticles","IsoTaus"),
                                      jets = cms.InputTag("rawSLHCL1ExtraParticles","Jets"),
                                      ##How to calibrate  
                                      ##Scale factor = MC/RAW = a+b |eta| +c|eta|^2 
                                      ##Give the coeffs for egamma and taus 
                                      ##So you need to fit the eta correction with a  
                                      ##parabola and add the coeffs here
                                      ## Same as we do for RCT Calibration
                                      eGammaCoefficients = cms.vdouble(1.0,0.0,0.0),
                                      tauCoefficients    = cms.vdouble(1.0,0.0,0.0)
)




