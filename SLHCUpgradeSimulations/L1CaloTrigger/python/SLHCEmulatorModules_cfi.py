import FWCore.ParameterSet.Config as cms

L1CaloTowerProducer = cms.EDProducer("L1CaloTowerProducer",
    ECALDigis = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    HCALDigis = cms.InputTag("simHcalTriggerPrimitiveDigis")
)

L1CaloClusterProducer = cms.EDProducer("L1CaloClusterProducer",
    L1Towers = cms.InputTag("L1CaloTowerProducer")

)

L1CaloClusterFilter = cms.EDProducer("L1CaloClusterFilter",
    CrudeClusters = cms.InputTag("L1CaloClusterProducer")
)

L1JetProducer = cms.EDProducer("L1JetMaker",
    FilteredClusters = cms.InputTag("L1CaloClusterFilter","FilteredClusters")
)

L1ExtraMaker = cms.EDProducer("L1ExtraMaker",
    Clusters = cms.InputTag("L1CaloClusterFilter","ParticleClusters"),
    Jets = cms.InputTag("L1JetProducer"),
    NObjects = cms.int32(8)   
)





