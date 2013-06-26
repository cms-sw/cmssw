import FWCore.ParameterSet.Config as cms

tccFlatToDigi = cms.EDProducer("EcalFEtoDigi",
    FileEventOffset = cms.untracked.int32(0),
    UseIdentityLUT = cms.untracked.bool(False),
    SuperModuleId = cms.untracked.int32(-1),
    debugPrintFlag = cms.untracked.bool(False),
    FlatBaseName = cms.untracked.string('ecal_tcc_')
)



