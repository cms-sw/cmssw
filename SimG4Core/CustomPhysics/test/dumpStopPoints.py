import FWCore.ParameterSet.Config as cms

process = cms.Process('SCAN')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:PYTHIA6_Exotica_HSCP_gluino300_cfg_py_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root')
)

process.rhStopDump = cms.EDAnalyzer (
    "RHStopDump",
    stoppedFile = cms.string("stoppedPoint.txt")
    )

process.rhStopDumpstep = cms.Path (process. rhStopDump)
process.shadule = cms.Schedule(process.rhStopDumpstep)
