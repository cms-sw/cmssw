import FWCore.ParameterSet.Config as cms

process = cms.Process("Castor")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("Configuration.StandardSequences.Generator_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("SimCalorimetry.CastorSim.castordigi_cfi")

process.load("RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi")


process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.load('RecoLocalCalo.Castor.Castor_cff')
process.castor_db_producer = cms.ESProducer("CastorDbProducer")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
)

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.RandomNumberGeneratorService.theSource.initialSeed = 15298183
process.RandomNumberGeneratorService.generator.initialSeed = 12503027
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 11120000
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 151
process.RandomNumberGeneratorService.mix.initialSeed = 14575
process.RandomNumberGeneratorService.simCastorDigis.initialSeed = 26



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.load("Configuration.Generator.QCD_Pt_15_20_cfi")

process.es_pool = cms.ESSource( "PoolDBESSource",
     process.CondDBSetup,
     timetype = cms.string('runnumber'),
#   connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierPrep/CMS_COND_30X_HCAL'),
    connect = cms.string('sqlite_file:testExample.db'),
     authenticationMethod = cms.untracked.uint32(0),
     toGet = cms.VPSet(
         cms.PSet(
             record = cms.string('CastorPedestalsRcd'),
             tag = cms.string('castor_pedestals_v1.0_test')
             ),
         cms.PSet(
             record = cms.string('CastorPedestalWidthsRcd'),
             tag = cms.string('castor_widths_v1.0_test')
             ),
         cms.PSet(
             record = cms.string('CastorGainsRcd'),
             tag = cms.string('castor_gains_v1.0_test')
             ),
         cms.PSet(
             record = cms.string('CastorQIEDataRcd'),
             tag = cms.string('castor_qie_v1.0_test')
             ),
         cms.PSet(
             record = cms.string('CastorElectronicsMapRcd'),
             tag = cms.string('castor_emap_v1.0_test')
              ),
         cms.PSet(
             record = cms.string('CastorChannelQualityRcd'),
             tag = cms.string('castor_channelstatus_v1.0_test')

             )
	     
     )
)
process.es_hardcode = cms.ESSource("CastorHardcodeCalibrations",
     toGet = cms.untracked.vstring('GainWidths')
 )


process.CaloSD = cms.PSet(
    DetailedTiming = cms.bool(False),
    EminTrack = cms.double(1.0),
    Verbosity = cms.int32(0),
    UseMap = cms.bool(True),
    CheckHits = cms.int32(25)
)



process.common_maximum_timex = cms.PSet( # need to be localy redefined
   MaxTrackTime  = cms.double(500.0),  # need to be localy redefined
   MaxTimeNames  = cms.vstring(), # need to be localy redefined
   MaxTrackTimes = cms.vdouble()  # need to be localy redefined
)


process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('castor.root')
)

process.p1 = cms.Path(process.generator*process.pgen*process.VtxSmeared*process.g4SimHits*process.mix*process.simCastorDigis*process.castorreco*process.CastorFullReco)
process.outpath=cms.EndPath(process.o1)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.DefaultCutValue = 10. 

process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.UseMagneticField = False

process.g4SimHits.CaloTrkProcessing.TestBeam = True

process.g4SimHits.StackingAction = cms.PSet(
   process.common_heavy_suppression,
   process.common_maximum_timex,        # need to be localy redefined
   TrackNeutrino = cms.bool(False),
   KillHeavy     = cms.bool(False),
   SaveFirstLevelSecondary = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInCalo    = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInMuon    = cms.untracked.bool(True)
)

process.g4SimHits.SteppingAction = cms.PSet(
   process.common_maximum_timex, # need to be localy redefined
   KillBeamPipe            = cms.bool(True),
   CriticalEnergyForVacuum = cms.double(2.0),
   CriticalDensity         = cms.double(1e-15),
   EkinNames               = cms.vstring(),
   EkinThresholds          = cms.vdouble(),
   EkinParticles           = cms.vstring(),
   Verbosity               = cms.untracked.int32(0)
)






