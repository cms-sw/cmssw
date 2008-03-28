import FWCore.ParameterSet.Config as cms

hscpFilter = cms.EDFilter("HSCPFilter",
    SingleMuPtMin = cms.double(20.0),
    PMin1 = cms.double(10.0),
    PMin3 = cms.double(200.0),
    PMin2 = cms.double(50.0),
    DoubleMuPtMin = cms.double(10.0),
    DeDxMin1 = cms.double(5.0),
    DeDxMin3 = cms.double(0.0),
    DeDxMin2 = cms.double(3.5)
)


