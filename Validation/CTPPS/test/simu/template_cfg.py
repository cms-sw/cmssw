import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_$ERA_cff import *
process = cms.Process('CTPPSTest', $ERA)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Validation.CTPPS.ctppsLHCInfoPlotter_cfi')
process.load('Configuration.Generator.randomXiThetaGunProducer_cfi')
process.load("CondCore.CondDB.CondDB_cfi")

# minimal logger settings
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# particle generator
process.generator.xi_max = 0.25
process.generator.theta_x_sigma = 60.e-6
process.generator.theta_y_sigma = 60.e-6

# default source
process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
)

process.CondDB.connect = 'frontier://FrontierProd/CMS_CONDITIONS'
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CTPPSPixelAnalysisMaskRcd'),
        tag = cms.string("CTPPSPixelAnalysisMask_Run3_v1_hlt")
    ))
)

# random seeds
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.PSet(initialSeed = cms.untracked.uint32(98765)),
    generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
    beamDivergenceVtxGenerator = cms.PSet(initialSeed = cms.untracked.uint32(3849)),
    ppsDirectProtonSimulation = cms.PSet(initialSeed = cms.untracked.uint32(4981))
)

# number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(int($N_EVENTS))
)

# LHCInfo plotter
process.ctppsLHCInfoPlotter.outputFile = "$OUT_LHCINFO"

# track distribution plotter
process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    outputFile = cms.string("$OUT_TRACKS")
)

# reconstruction plotter
process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons", "singleRP"),
    tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons", "multiRP"),
    outputFile = cms.string("$OUT_PROTONS")
)

process.generation = cms.Path(process.generator)

process.validation = cms.Path(
    process.ctppsLHCInfoPlotter
    * process.ctppsTrackDistributionPlotter
    * process.ctppsProtonReconstructionPlotter
)

# processing path
process.schedule = cms.Schedule(
    process.generation,
    process.validation
)

from SimPPS.Configuration.Utils import setupPPSDirectSim
setupPPSDirectSim(process)

process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetX45 = 0.
process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetY45 = 0.
process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetZ45 = 0.
process.source.numberEventsInLuminosityBlock = process.ctppsCompositeESSource.generateEveryNEvents
process.ctppsTrackDistributionPlotter.rpId_45_F = process.rpIds.rp_45_F
process.ctppsTrackDistributionPlotter.rpId_45_N = process.rpIds.rp_45_N
process.ctppsTrackDistributionPlotter.rpId_56_N = process.rpIds.rp_56_N
process.ctppsTrackDistributionPlotter.rpId_56_F = process.rpIds.rp_56_F
process.ctppsProtonReconstructionPlotter.rpId_45_F = process.rpIds.rp_45_F
process.ctppsProtonReconstructionPlotter.rpId_45_N = process.rpIds.rp_45_N
process.ctppsProtonReconstructionPlotter.rpId_56_N = process.rpIds.rp_56_N
process.ctppsProtonReconstructionPlotter.rpId_56_F = process.rpIds.rp_56_F
