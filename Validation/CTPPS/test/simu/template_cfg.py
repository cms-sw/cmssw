import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_$ERA_cff import *
process = cms.Process('CTPPSTest', $ERA)

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'frontier://FrontierProd/CMS_CONDITIONS'
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CTPPSPixelAnalysisMaskRcd'),
        tag = cms.string("CTPPSPixelAnalysisMask_Run3_v1_hlt")
    ))
)

# minimal logger settings
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# global tag
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '120X_mcRun3_2021_realistic_v6', '')
#process.load('Geometry.VeryForwardGeometry.geometryRPFromDB_cfi')

# load config
process.load('SimPPS.Configuration.ppsDirectSim_cff')
process.load('RecoPPS.Configuration.recoCTPPS_cff')
process.load('RecoPPS.ProtonReconstruction.ctppsProtons_cff')

# default source
process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = process.ctppsCompositeESSource.generateEveryNEvents
)

# particle generator
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
from Configuration.Generator.randomXiThetaGunProducer_cfi import generator as _gen
process.generator = _gen.clone(
    xi_max = 0.25,
    theta_x_sigma = 60.e-6,
    theta_y_sigma = 60.e-6
)

# beam smearing
process.load('IOMC.EventVertexGenerators.beamDivergenceVtxGenerator_cfi')

# random seeds
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.PSet(initialSeed = cms.untracked.uint32(98765)),
    generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
    beamDivergenceVtxGenerator = cms.PSet(initialSeed = cms.untracked.uint32(3849)),
    ppsDirectProtonSimulation = cms.PSet(initialSeed = cms.untracked.uint32(4981))
)

from SimPPS.DirectSimProducer.profile_base_cff import matchDirectSimOutputs
matchDirectSimOutputs(process)

# number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(int($N_EVENTS))
)

# LHCInfo plotter
process.load("Validation.CTPPS.ctppsLHCInfoPlotter_cfi")
process.ctppsLHCInfoPlotter.outputFile = "$OUT_LHCINFO"

# track distribution plotter
process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),

    rpId_45_F = process.rpIds.rp_45_F,
    rpId_45_N = process.rpIds.rp_45_N,
    rpId_56_N = process.rpIds.rp_56_N,
    rpId_56_F = process.rpIds.rp_56_F,

    outputFile = cms.string("$OUT_TRACKS")
)

# reconstruction plotter
process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons", "singleRP"),
    tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons", "multiRP"),

    rpId_45_F = process.rpIds.rp_45_F,
    rpId_45_N = process.rpIds.rp_45_N,
    rpId_56_N = process.rpIds.rp_56_N,
    rpId_56_F = process.rpIds.rp_56_F,

    outputFile = cms.string("$OUT_PROTONS")
)

# processing path
process.p = cms.Path(
    process.generator
    * process.beamDivergenceVtxGenerator
    * process.ppsDirectProtonSimulation

    * process.recoCTPPS
    * process.ctppsProtons

    * process.ctppsLHCInfoPlotter
    * process.ctppsTrackDistributionPlotter
    * process.ctppsProtonReconstructionPlotter
)
