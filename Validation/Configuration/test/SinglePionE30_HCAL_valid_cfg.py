import FWCore.ParameterSet.Config as cms

process = cms.Process("ValidationSimDigiChain")
#
# Master configuration for the magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# Geometry master configuration
#
process.load("Configuration.StandardSequences.GeometryHCAL_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("CalibCalorimetry.Configuration.Hcal_FakeConditions_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.EventContent.EventContent_cff")

# re-create the CrossingFrame
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

# HCAL validation sequences
#
process.load("Validation.Configuration.hcalSimValid_cff")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    catalog = cms.untracked.string('PoolFileCatalog.xml'),
    fileNames = cms.untracked.vstring('file:PI-_30_ALL.root')
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMDIGIEventContent,
    fileName = cms.untracked.string('PI-_30_ALL_valid.root')
)

process.p0 = cms.Path(process.mix)
process.p2 = cms.Path(process.hcalSimValid)
process.p3 = cms.Path(process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.p0,process.p2,process.p3,process.outpath)

process.mix.playback = True

