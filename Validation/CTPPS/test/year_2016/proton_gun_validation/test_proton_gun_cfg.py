import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
process = cms.Process('CTPPSFastSimulation', ctpps_2016)

# minimal logger settings
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# number of events
process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(280000)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# particle-data table
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# provide LHC info
# TODO: remove this line once data are available in CondDB
process.load("CalibPPS.ESProducers.ctppsLHCInfo_cff")

# particle generator
process.load("Validation.CTPPS.year_2016.randomXiThetaGunProducer_cfi")

# random seeds
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.PSet(initialSeed =cms.untracked.uint32(98765)),
    generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
)

# plotter
process.ctppsHepMCDistributionPlotter = cms.EDAnalyzer("CTPPSHepMCDistributionPlotter",
    tagHepMC = cms.InputTag("generator", "unsmeared"),
    lhcInfoLabel = process.ctppsLHCInfoESSource_2016.label, # TODO: replace with "" once data loaded from DB
    outputFile = cms.string("test_proton_gun.root")
)

# processing path
process.p = cms.Path(
    process.generator
    * process.ctppsHepMCDistributionPlotter
)
