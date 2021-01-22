import FWCore.ParameterSet.Config as cms

# Use this object to make changes when different configurations are active

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
    nPhi = cms.int32(72),
    hcalRegion = cms.int32(2)
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
    nPhi = cms.int32(48),
    includeBadChambers = cms.bool(False),
    includeGEM = cms.bool(False),
    includeME0 = cms.bool(False)
)

# If running in Run 2, include bad chambers
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( muonDetIdAssociator, includeBadChambers = True )

# include GEM & ME0 for phase2
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( muonDetIdAssociator, includeGEM = True )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify( muonDetIdAssociator, includeME0 = True )
phase2_muon.toModify( hcalDetIdAssociator, hcalRegion = 1 )
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toModify( muonDetIdAssociator, includeME0 = False )

preshowerDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('PreshowerDetIdAssociator'),
    etaBinSize = cms.double(0.1),
    nEta = cms.int32(60),
    nPhi = cms.int32(30)
)
