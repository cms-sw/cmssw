import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalTBH4GenSimDigi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    saveFileName = cms.untracked.string(''),
    generator = cms.PSet(
       initialSeed = cms.untracked.uint32(123456789),      
        engineName = cms.untracked.string('HepJamesRandom') 
    ),                                                      
    VtxSmeared = cms.PSet(
        initialSeed = cms.untracked.uint32(98765432),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    g4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    SimEcalTBG4Object = cms.PSet(
        initialSeed = cms.untracked.uint32(12),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    mix = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

#Geometry

process.load("Geometry.EcalTestBeam.TBH4_2007_GeometryIdeal_cfi")

# No magnetic field

process.load("MagneticField.Engine.uniformMagneticField_cfi")

process.source = cms.Source("EmptySource")

# defines common_beam_direction_parameters
process.load("SimG4CMS.EcalTestBeam.ee_PositionParticleGun_cff")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        process.common_beam_direction_parameters,
        PartID = cms.vint32(11),
        MinE = cms.double(119.99),
        MaxE = cms.double(120.01)
    ),
    Verbosity = cms.untracked.int32(1), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single electron'),
    AddAntiParticle = cms.bool(False),
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

#
# this module takes input in the units of *cm* and *radian*!!!
#

process.VtxSmeared = cms.EDProducer("BeamProfileVtxGenerator",
    process.common_beam_direction_parameters,
    VtxSmearedCommon,
    BeamSigmaX = cms.double(2.4),
    BeamSigmaY = cms.double(2.4),
    GaussianProfile = cms.bool(False),
    BinX = cms.int32(50),
    BinY = cms.int32(50),
    File       = cms.string('beam.profile'),
    UseFile    = cms.bool(False),
    TimeOffset = cms.double(0.)                      
)

# Geant4, ECAL test beam specific OscarProducer configuration

process.load("SimG4Core.Application.g4SimHits_cfi")

process.g4SimHits.UseMagneticField = cms.bool(False)
process.g4SimHits.Physics.DefaultCutValue = 1.
process.g4SimHits.NonBeamEvent = cms.bool(True)
process.g4SimHits.Generator.HepMCProductLabel = cms.string('generatorSmeared')
process.g4SimHits.Generator.ApplyPCuts = cms.bool(False)
process.g4SimHits.Generator.ApplyEtaCuts = cms.bool(True)
process.g4SimHits.Generator.ApplyPhiCuts = cms.bool(False)
process.g4SimHits.Generator.MaxEtaCut = cms.double(2.5)
process.g4SimHits.Generator.MinEtaCut = cms.double(0.0)
process.g4SimHits.CaloSD.CorrectTOFBeam = cms.bool(True)
process.g4SimHits.CaloSD.BeamPosition = cms.double(-26733.5)
process.g4SimHits.CaloTrkProcessing.TestBeam = cms.bool(True)
process.g4SimHits.StackingAction.MaxTrackTime = cms.double(10000.)
process.g4SimHits.SteppingAction.MaxTrackTime = cms.double(10000.)
process.g4SimHits.CaloSD.TmaxHit = cms.double(10000.)
process.g4SimHits.CaloSD.TmaxHits = cms.vdouble(10000.,10000.,10000.,10000.,10000.)

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('EcalTBH4Trigger'),
    verbose = cms.untracked.bool(False),
    #IMPORTANT    #    #    #    #    #    #    #    # NUMBER OF EVENTS TO BE TRIGGERED 
    trigEvents = cms.untracked.int32(25)
))


# Test Beam ECAL specific MC info

process.SimEcalTBG4Object = cms.EDProducer("EcalTBMCInfoProducer",
    process.common_beam_direction_parameters,
    CrystalMapFile = cms.FileInPath('Geometry/EcalTestBeam/data/BarrelSM1CrystalCenterElectron120GeV.dat'),
    moduleLabelVtx = cms.untracked.string('generatorSmeared')
)

# Test Beam ECAL hodoscope raw data simulation

process.SimEcalTBHodoscope = cms.EDProducer("TBHodoActiveVolumeRawInfoProducer")

# Test Beam ECAL Event header filling

process.SimEcalEventHeader = cms.EDProducer("FakeTBEventHeaderProducer",
    EcalTBInfoLabel = cms.untracked.string('SimEcalTBG4Object')
)

# Digitization

# no pileup

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

# fake TB conditions

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetrieverTB_cfi")

# Test beam unsuppressed digis

process.load("SimCalorimetry.EcalTestBeam.ecaldigi_testbeam_cfi")
process.mix.digitizers.ecal.doReadout = False

# Output

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
        'drop PSimHits_g4SimHits_*_Sim', 
        'keep PCaloHits_g4SimHits_EcalHitsEE_Sim', 
        'keep PCaloHits_g4SimHits_CaloHitsTk_Sim', 
        'keep PCaloHits_g4SimHits_EcalTBH4BeamHits_Sim'),
    fileName = cms.untracked.string('ECALH4TB_detsim_digi.root')
)

# sequences

process.doSimHits = cms.Sequence(process.ProductionFilterSequence*process.VtxSmeared*process.g4SimHits)
process.doSimTB = cms.Sequence(process.SimEcalTBG4Object*process.SimEcalTBHodoscope*process.SimEcalEventHeader)
process.doEcalDigis = cms.Sequence(process.mix)
process.p1 = cms.Path(process.doSimHits*process.doSimTB*process.doEcalDigis)
process.outpath = cms.EndPath(process.output)


# modify the default behavior of the MessageLogger
    
process.MessageLogger.destinations=cms.untracked.vstring('cout'
                                                         ,'cerr'
                                                         ,'G4msg'
                                                         )
process.MessageLogger.categories=cms.untracked.vstring('FwkJob'
                                                       ,'FwkReport'
                                                       ,'FwkSummary'
                                                       ,'Root_NoDictionary'
                                                       ,'TimeReport'
                                                       ,'TimeModule'
                                                       ,'TimeEvent'
                                                       ,'MemoryCheck'
                                                       ,'PhysicsList'
                                                       ,'G4cout'
                                                       ,'G4cerr'
                                                       ,'BeamProfileVtxGenerator'
                                                       )

process.MessageLogger.debugModules = cms.untracked.vstring('g4SimHits','generatorSmeared')

#Configuring the G4msg.log output
process.MessageLogger.G4msg =  cms.untracked.PSet(
    noTimeStamps = cms.untracked.bool(True)
    #First eliminate unneeded output
    ,threshold = cms.untracked.string('INFO')
    #,DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0))
    ,INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
    ,FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(0))
    ,FwkSummary = cms.untracked.PSet(limit = cms.untracked.int32(0))
    ,Root_NoDictionary = cms.untracked.PSet(limit = cms.untracked.int32(0))
    ,FwkJob = cms.untracked.PSet(limit = cms.untracked.int32(0))
    ,TimeReport = cms.untracked.PSet(limit = cms.untracked.int32(0))
    ,TimeModule = cms.untracked.PSet(limit = cms.untracked.int32(0))
    ,TimeEvent = cms.untracked.PSet(limit = cms.untracked.int32(0))
    ,MemoryCheck = cms.untracked.PSet(limit = cms.untracked.int32(0))
    #TimeModule, TimeEvent, TimeReport are written to LogAsbolute instead of LogInfo with a category
    #so they cannot be eliminated from any destination (!) unless one uses the summaryOnly option
    #in the Timing Service... at the price of silencing the output needed for the TimingReport profiling
    #
    #Then add the wanted ones:
    ,PhysicsList = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    ,G4cout = cms.untracked.PSet(limit = cms.untracked.int32(99999))
    ,G4cerr = cms.untracked.PSet(limit = cms.untracked.int32(99999))
    ,BeamProfileVtxGenerator = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    )

#Add these 3 lines to put back the summary for timing information at the end of the logfile
#(needed for TimeReport report)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
#process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
#process.MessageLogger.debugModules = cms.untracked.vstring('*')


#process.load("Validation.Performance.TimeMemoryG4Info")


#process.g4SimHits.G4Commands = cms.vstring('/tracking/verbose 1')
