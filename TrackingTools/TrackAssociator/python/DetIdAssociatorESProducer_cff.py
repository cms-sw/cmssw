import FWCore.ParameterSet.Config as cms

ecalDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('EcalDetIdAssociator'),
    etaBinSize = cms.double(0.02),
    nEta = cms.int32(300),
    nPhi = cms.int32(360)
)

hcalDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('HcalDetIdAssociator'),
    etaBinSize = cms.double(0.087),
    nEta = cms.int32(70),
    nPhi = cms.int32(72)
)

hoDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('HODetIdAssociator'),
    etaBinSize = cms.double(0.087),
    nEta = cms.int32(30),
    nPhi = cms.int32(72)
)

caloDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('CaloDetIdAssociator'),
    etaBinSize = cms.double(0.087),
    nEta = cms.int32(70),
    nPhi = cms.int32(72)
)

muonDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('MuonDetIdAssociator'),
    etaBinSize = cms.double(0.125),
    nEta = cms.int32(48),
    nPhi = cms.int32(48)
)


