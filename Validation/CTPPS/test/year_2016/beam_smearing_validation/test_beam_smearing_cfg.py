import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('CTPPSTestBeamSmearing', eras.ctpps_2016)

# minimal logger settings
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# number of events
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# particle-data table
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# supply LHC info
process.load("Validation.CTPPS.year_2016.ctppsLHCInfoESSource_cfi")

# supply optics
process.load("Validation.CTPPS.year_2016.ctppsOpticalFunctionsESSource_cfi")

# supply beam parameters
process.load("Validation.CTPPS.year_2016.ctppsBeamParametersESSource_cfi")

# particle generator
process.load("Validation.CTPPS.year_2016.randomXiThetaGunProducer_cfi")

# random seeds
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.PSet(initialSeed =cms.untracked.uint32(98765)),
    generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
    beamDivergenceVtxGenerator = cms.PSet(initialSeed =cms.untracked.uint32(3849))
)

# beam-smearing
process.load("IOMC.EventVertexGenerators.beamDivergenceVtxGenerator_cfi")

# beam-smearing validation
process.ctppsBeamSmearingValidator = cms.EDAnalyzer("CTPPSBeamSmearingValidator",
  tagBeforeSmearing = cms.InputTag("generator", "unsmeared"),
  tagAfterSmearing = cms.InputTag("beamDivergenceVtxGenerator"),
  outputFile = cms.string("test_beam_smearing.root")
)

# processing path
process.p = cms.Path(
    process.generator
    * process.beamDivergenceVtxGenerator
    * process.ctppsBeamSmearingValidator
)
