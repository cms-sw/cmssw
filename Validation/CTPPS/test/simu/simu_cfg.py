import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_$ERA_cff import *
process = cms.Process('CTPPSTest', $ERA)

# load configs
import Validation.CTPPS.simu_config.year_$CONFIG_cff
process.load("Validation.CTPPS.simu_config.year_$CONFIG_cff")

#set constant xangle/beta* for whole 2016
#config_2016.UseConstantXangleBetaStar(process,140,0.3)


#set weights
#process.profile_2017_postTS2.L_i=1
#process.profile_2017_preTS2.L_i=10

process.ctppsCompositeESSource.generateEveryNEvents=1000
process.source.numberEventsInLuminosityBlock=1000

process.Timing=cms.Service("Timing",
	summaryOnly=cms.untracked.bool(True),
	useJobReport=cms.untracked.bool(True)				
)
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
  input = cms.untracked.int32(100000)
)

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

  association_cuts_45 = process.ctppsProtons.association_cuts_45,
  association_cuts_56 = process.ctppsProtons.association_cuts_56,

  outputFile = cms.string("$OUT_PROTONS")
)


# processing path
process.p = cms.Path(
  process.generator
  * process.beamDivergenceVtxGenerator
  * process.ctppsDirectProtonSimulation

  * process.reco_local
  * process.ctppsProtons

  * process.ctppsTrackDistributionPlotter
  * process.ctppsProtonReconstructionPlotter
)
