import FWCore.ParameterSet.Config as cms

EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions",
    calDigi = cms.untracked.PSet(
        enableOverFlowEx = cms.untracked.bool(True),
        enableDivByZeroEx = cms.untracked.bool(True),
        enableInvalidEx = cms.untracked.bool(True),
        enableUnderFlowEx = cms.untracked.bool(True)
    ),
    moduleNames = cms.untracked.vstring('default', 
        'VtxSmeared', 
        'g4SimHits', 
        'mix', 
        'trDigi', 
        'calDigi', 
        'muonDigi'),
    trDigi = cms.untracked.PSet(
        enableOverFlowEx = cms.untracked.bool(True),
        enableDivByZeroEx = cms.untracked.bool(True),
        enableInvalidEx = cms.untracked.bool(True),
        enableUnderFlowEx = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(
        enableOverFlowEx = cms.untracked.bool(False),
        enableDivByZeroEx = cms.untracked.bool(False),
        enableInvalidEx = cms.untracked.bool(False),
        enableUnderFlowEx = cms.untracked.bool(False)
    ),
    mix = cms.untracked.PSet(
        enableOverFlowEx = cms.untracked.bool(True),
        enableDivByZeroEx = cms.untracked.bool(True),
        enableInvalidEx = cms.untracked.bool(True),
        enableUnderFlowEx = cms.untracked.bool(True)
    ),
    VtxSmeared = cms.untracked.PSet(
        enableOverFlowEx = cms.untracked.bool(True),
        enableDivByZeroEx = cms.untracked.bool(True),
        enableInvalidEx = cms.untracked.bool(True),
        enableUnderFlowEx = cms.untracked.bool(True)
    ),
    setPrecisionDouble = cms.untracked.bool(True),
    reportSettings = cms.untracked.bool(True),
    muonDigi = cms.untracked.PSet(
        enableOverFlowEx = cms.untracked.bool(True),
        enableDivByZeroEx = cms.untracked.bool(True),
        enableInvalidEx = cms.untracked.bool(True),
        enableUnderFlowEx = cms.untracked.bool(True)
    ),
    g4SimHits = cms.untracked.PSet(
        enableOverFlowEx = cms.untracked.bool(False),
        enableDivByZeroEx = cms.untracked.bool(False),
        enableInvalidEx = cms.untracked.bool(False),
        enableUnderFlowEx = cms.untracked.bool(False)
    )
)


