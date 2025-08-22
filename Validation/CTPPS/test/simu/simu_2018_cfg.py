import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import *
process = cms.Process('CTPPSTest', Run2_2018)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Validation.CTPPS.ctppsLHCInfoPlotter_cfi')
process.load('Configuration.Generator.randomXiThetaGunProducer_cfi')
process.load("CondCore.CondDB.CondDB_cfi")
# process.load('SimPPS.DirectSimProducer.ppsDirectProtonSimulation_cfi') 
process.load('SimPPS.DirectSimProducer.ctppsGregDucer_cfi')

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
        tag = cms.string("CTPPSPixelAnalysisMask_Run3_v1_hlt"))
        ))

# random seeds
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.PSet(initialSeed = cms.untracked.uint32(98765)),
    generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
    beamDivergenceVtxGenerator = cms.PSet(initialSeed = cms.untracked.uint32(3849)),
    ppsDirectProtonSimulation = cms.PSet(initialSeed = cms.untracked.uint32(4981))
)

# number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(int(1E5))
)

# LHCInfo plotter
process.ctppsLHCInfoPlotter.outputFile = "simu_2018_lhcInfo.root"

# track distribution plotter
process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    outputFile = cms.string("simu_2018_tracks.root")
)

# reconstruction plotter
process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons", "singleRP"),
    tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons", "multiRP"),
    outputFile = cms.string("simu_2018_protons.root")
)



# Greg plotter 1 - Unfiltered
process.ctppsGregPlotter = cms.EDAnalyzer("CTPPSGregPlotter",
    tagTracks = cms.InputTag("GenParticles"),
    outputFile = cms.string("simu_2018_Greg.root")#,
)

# Greg producer 1 - Calibration
process.ctppsGregDucer1 = cms.EDProducer("CTPPSGregDucer",
    tagTracks = cms.InputTag("GenParticlesNew"),
    hepMCTag = cms.InputTag("generator", "unsmeared"),
    # filename = cms.string("/afs/cern.ch/user/g/gjedrzej/private/mainTask/CMSSW_15_0_11/src/SimPPS/DirectSimProducer/cutFiles/thetaphilimits_-160urad_18cm_60cm_calib-nodet-xrphd-64bins.out")
    filename = cms.string("../../../../SimPPS/DirectSimProducer/cutFiles/thetaphilimits_-160urad_18cm_60cm_calib-nodet-xrphd-64bins.out")
)

# Greg plotter 2 - Calibration
process.ctppsGregPlotter2 = cms.EDAnalyzer("CTPPSGregPlotter",
    tagTracks = cms.InputTag("ctppsGregDucer1", "selectedProtons"),
    hepMCTag = cms.InputTag("ctppsGregDucer1", "selectedProtons"),
    outputFile = cms.string("simu_2018_64binsCalib.root")

)

# Greg producer 2 - Physics
process.ctppsGregDucer2 = cms.EDProducer("CTPPSGregDucer",
    tagTracks = cms.InputTag("GenParticlesNew"),
    hepMCTag = cms.InputTag("generator", "unsmeared"),
    # filename = cms.string("/afs/cern.ch/user/g/gjedrzej/private/mainTask/CMSSW_15_0_11/src/SimPPS/DirectSimProducer/cutFiles/thetaphilimits_-160urad_18cm_60cm_phys-nodet-xrphd-64bins.out")
    filename = cms.string("../../../../SimPPS/DirectSimProducer/cutFiles/thetaphilimits_-160urad_18cm_60cm_phys-nodet-xrphd-64bins.out")

)


# Greg plotter 3 - Physics
process.ctppsGregPlotter3 = cms.EDAnalyzer("CTPPSGregPlotter",
    tagTracks = cms.InputTag("ctppsGregDucer2", "selectedProtons"),
    hepMCTag = cms.InputTag("ctppsGregDucer2", "selectedProtons"),
    outputFile = cms.string("simu_2018_64binsPhys.root")
)

process.generation = cms.Path(process.generator)


process.validation = cms.Path(
    process.ctppsLHCInfoPlotter
    * process.ctppsTrackDistributionPlotter
    * process.ctppsProtonReconstructionPlotter
    * process.ctppsGregPlotter 
)

# Calibration
process.cutAndValidate = cms.Path(
    process.ctppsGregDucer1 * process.ctppsGregPlotter2
)

# Physics
process.cutAndValidate2 = cms.Path(
    process.ctppsGregDucer2 * process.ctppsGregPlotter3)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("data_output.root"),
    # Keep only the new products
    outputCommands = cms.untracked.vstring(
        "keep *",
        # "drop *_*_*_RECO"
    )
)

process.end_path = cms.EndPath(
  process.output
)

# processing path
process.schedule = cms.Schedule(
    process.generation,
    process.validation,
    process.cutAndValidate,
    process.cutAndValidate2,
    process.end_path
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


