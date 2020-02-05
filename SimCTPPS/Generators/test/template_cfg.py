import FWCore.ParameterSet.Config as cms

# load config
from Validation.CTPPS.simu_config.year_$config_cff import *

process = cms.Process('CTPPSDirectSimulation', era)

process.load("Validation.CTPPS.simu_config.year_$config_cff")
UseCrossingAngle($xangle, process)

# minimal logger settings
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(int($n_events))
)

# redefine particle generator
process.load("SimCTPPS.Generators.PPXZGenerator_cfi")
process.generator.verbosity = 0
process.generator.m_X = $mass
process.generator.m_XZ_min = $mass + 100
process.generator.m_X_pr1 = $mass - 100
process.generator.decayX = True

# distribution plotter
process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
  tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
  outputFile = cms.string("$JOB_DIR/output_shape_smear.root")
)

# acceptance plotter
process.ctppsAcceptancePlotter = cms.EDAnalyzer("CTPPSAcceptancePlotter",
  tagHepMC = cms.InputTag("generator", "unsmeared"),
  tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),

  rpId_45_F = process.rpIds.rp_45_F,
  rpId_45_N = process.rpIds.rp_45_N,
  rpId_56_N = process.rpIds.rp_56_N,
  rpId_56_F = process.rpIds.rp_56_F,

  outputFile = cms.string("$JOB_DIR/acceptance.root")
)

# generator plots
process.load("SimCTPPS.Generators.PPXZGeneratorValidation_cfi")
process.ppxzGeneratorValidation.tagHepMC = cms.InputTag("generator", "unsmeared")
process.ppxzGeneratorValidation.tagRecoTracks = cms.InputTag("ctppsLocalTrackLiteProducer")
process.ppxzGeneratorValidation.tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons", "singleRP")
process.ppxzGeneratorValidation.tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons", "multiRP")
process.ppxzGeneratorValidation.referenceRPDecId_45 = process.rpIds.rp_45_F
process.ppxzGeneratorValidation.referenceRPDecId_56 = process.rpIds.rp_56_F
process.ppxzGeneratorValidation.outputFile = "$JOB_DIR/ppxzGeneratorValidation.root"

# processing path
process.p = cms.Path(
  process.generator
  * process.beamDivergenceVtxGenerator
  * process.ctppsDirectProtonSimulation

  * process.reco_local
  * process.ctppsProtons

  * process.ctppsTrackDistributionPlotter
  * process.ctppsAcceptancePlotter
  * process.ppxzGeneratorValidation
)


# output configuration
process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("file://ntuple.root"),
  splitLevel = cms.untracked.int32(0),
  eventAutoFlushCompressedSize=cms.untracked.int32(-900),
  compressionAlgorithm=cms.untracked.string("LZMA"),
  compressionLevel=cms.untracked.int32(9),
  outputCommands = cms.untracked.vstring(
    'drop *',
    'keep edmHepMCProduct_*_*_*',
    'keep CTPPSLocalTrackLites_*_*_*',
    'keep recoForwardProtons_*_*_*'
  )
)

if ($save_ntuples):
  process.outpath = cms.EndPath(process.output)

def UseSettingsZ():
  pass

def UseSettingsGamma():
  process.generator.m_Z_mean = 0
  process.generator.m_Z_gamma = 0
  process.generator.m_XZ_min = $mass + 1E-6
  process.generator.m_X_pr1 = $mass - 100
  process.generator.p_T_Z_min = 80

$settings_function()
