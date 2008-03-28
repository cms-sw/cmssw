import FWCore.ParameterSet.Config as cms

# Test Beam ECAL specific MC info
simCaloTB = cms.EDProducer("EcalTBMCInfoProducer",
    common_beam_direction_parameters,
    CrystalMapFile = cms.FileInPath('Geometry/EcalTestBeam/data/BarrelSM1CrystalCenterElectron120GeV.dat'),
    moduleLabelVtx = cms.untracked.string('source')
)

doSimTB = cms.Sequence(simCaloTB)

