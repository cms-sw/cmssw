import FWCore.ParameterSet.Config as cms
process = cms.Process("GeometryTest")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# no events to process
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

# retrieve the two plotters
process.load('Validation.CTPPS.opticalFunctionsPlotter_cfi')

# prepare the output file
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('optical_functions.root'),
    closeFileFast = cms.untracked.bool(True),
)

process.p = cms.Path(
    process.ctppsPlotOpticalFunctions_45
    * process.ctppsPlotOpticalFunctions_56
)
