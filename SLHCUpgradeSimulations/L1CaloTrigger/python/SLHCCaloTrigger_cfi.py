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


SLHCL1ExtraParticles = cms.EDProducer("L1ExtraTranslator",
                                  Clusters = cms.InputTag("L1CaloClusterIsolator"),
                                  Jets = cms.InputTag("L1CaloJetExpander"),
                                  NParticles = cms.int32(8),
                                  NJets      = cms.int32(12)
                              
)




