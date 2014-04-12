import FWCore.ParameterSet.Config as cms
import os

mySample = "g4"
mySection = "0"
myEvent = "100"
mySeed = "32789"
myEnergy = "20"

# condor_output_vtx50 -------------------------------------
# vertex sigma 5.0 : not much different at this time.
myParList = cms.vdouble(1.006, 1.0, 0.0, 1.82790e+00, 3.66237e+00, 0.965, 1.0)
myNameTag = mySample + "_" + myEnergy + "_" + mySection

process = cms.Process("TBSim")




process.load("FWCore.MessageLogger.MessageLogger_cfi")

    
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    saveFileName = cms.untracked.string(''),
    theSource = cms.PSet(
        initialSeed = cms.untracked.uint32(7824367),      
        engineName = cms.untracked.string('HepJamesRandom') 
    ),                                                      
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(int(mySeed)),      
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
    input = cms.untracked.int32(int(myEvent)*10)
)

#Geometry

#process.load("SimG4CMS.EcalTestBeam.crystal248_cff")
process.common_beam_direction_parameters = cms.PSet(
    BeamMeanY = cms.double(0.0),
    BeamMeanX = cms.double(0.0),
    MinEta = cms.double(0.221605),
    MaxEta = cms.double(0.221605),
    MinPhi = cms.double(0.0467487),
    MaxPhi = cms.double(0.0467487),
#    Psi    = cms.double(999.9),
    BeamPosition = cms.double(-26733.5)
)



#process.load("Geometry.EcalTestBeam.TBH4GeometryXML_cfi")
process.MuonNumberingInitialization = cms.ESProducer("MuonNumberingInitialization")
process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/EcalTestBeam/data/ebcon.xml', 
        'Geometry/EcalCommonData/data/ebrot.xml', 
        'Geometry/EcalTestBeam/data/eregalgo.xml', 
        'Geometry/EcalCommonData/data/ebalgo.xml', 
        'Geometry/EcalTestBeam/data/tbrot.xml',
        'Geometry/EcalTestBeam/data/TBH4.xml', 
        'Geometry/EcalTestBeam/data/TBH4ecalsens.xml', 
        'Geometry/HcalSimData/data/CaloUtil.xml', 
        #'Geometry/EcalSimData/data/EcalProdCuts.xml', 
        #'Geometry/EcalTestBeam/data/TBH4ProdCuts.xml',
        'SimG4Core/GFlash/TB/gflashTBH4ProdCuts.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('TBH4:OCMS')
)


process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.CaloGeometryBuilder.SelectedCalos = ['EcalBarrel']
#, 'EcalEndcap']


# No magnetic field

process.load("MagneticField.Engine.uniformMagneticField_cfi")

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        process.common_beam_direction_parameters,
        PartID = cms.vint32(11),
        MinE = cms.double(float(myEnergy)),
        MaxE = cms.double(float(myEnergy))
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

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
#    BeamSigmaX = cms.double(2.4),
#    BeamSigmaY = cms.double(2.4),
#    GaussianProfile = cms.bool(False),
    BeamSigmaX = cms.double(5.0),
    BeamSigmaY = cms.double(5.0),
    Psi             = cms.double(999.9),
    GaussianProfile = cms.bool(True),
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
process.g4SimHits.Generator.HepMCProductLabel = cms.string('generator')
process.g4SimHits.Generator.ApplyPCuts = cms.bool(False)
process.g4SimHits.Generator.ApplyEtaCuts = cms.bool(True)
process.g4SimHits.Generator.ApplyPhiCuts = cms.bool(False)
process.g4SimHits.Generator.MaxEtaCut = cms.double(1.5)
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
    trigEvents = cms.untracked.int32(int(myEvent))
))


# Test Beam ECAL specific MC info

process.SimEcalTBG4Object = cms.EDProducer("EcalTBMCInfoProducer",
    process.common_beam_direction_parameters,
    CrystalMapFile = cms.FileInPath('Geometry/EcalTestBeam/data/BarrelSM1CrystalCenterElectron120GeV.dat'),
    moduleLabelVtx = cms.untracked.string('generator')
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

process.load("CalibCalorimetry.EcalTrivialCondModules.ESTrivialCondRetriever_cfi")

# Test beam unsuppressed digis

process.load("SimCalorimetry.EcalTestBeam.ecaldigi_testbeam_cfi")

# local reco
process.load("Configuration.EcalTB.localReco_tbsim_cff")
 
# ntuplizer for TB data format
#process.treeProducerCalibSimul = cms.EDFilter("TreeProducerCalibSimul",
#    rootfile = cms.untracked.string("treeTB_"+myNameTag+".root"),
#    eventHeaderCollection = cms.string(''),
#    eventHeaderProducer = cms.string('SimEcalEventHeader'),
#    txtfile = cms.untracked.string("treeTB_"+myNameTag+".txt"),
#    EBRecHitCollection = cms.string('EcalRecHitsEB'),
#    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
#    xtalInBeam = cms.untracked.int32(248),
#    hodoRecInfoProducer = cms.string('ecalTBSimHodoscopeReconstructor'),
#    hodoRecInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
#    RecHitProducer = cms.string('ecalTBSimRecHit'),
#    tdcRecInfoProducer = cms.string('ecalTBSimTDCReconstructor')
#)

# turning on/off Gflash
if mySample == "gf":
    process.ecal_notCont_sim.EBs25notContainment = 1.0
    process.ecal_notCont_sim.EEs25notContainment = 1.0
    process.g4SimHits.Physics.type = 'SimG4Core/Physics/GFlash'
    process.g4SimHits.Physics.GFlash = cms.PSet(
        bField = cms.double(0.0),
        GflashEMShowerModel = cms.bool(True),
        GflashHadronShowerModel = cms.bool(True),
        GflashHistogram = cms.bool(True),
        GflashHistogramName = cms.string("gflash_histogram_"+myNameTag+".root"),
        GflashHadronPhysics = cms.string('QGSP_BERT'),
        GflashHcalOuter = cms.bool(True),
        GflashExportToFastSim = cms.bool(False),
        watcherOn = cms.bool(False),
        Verbosity = cms.untracked.int32(0),
        tuning_pList = myParList
    )

print "physics type : ", process.g4SimHits.Physics.type

# sequences
process.doSimHits = cms.Sequence(process.ProductionFilterSequence*process.VtxSmeared*process.g4SimHits)
process.doSimTB = cms.Sequence(process.SimEcalTBG4Object*process.SimEcalTBHodoscope*process.SimEcalEventHeader)
process.doEcalDigis = cms.Sequence(process.mix)
#process.p1 = cms.Path(process.doSimHits*process.doSimTB*process.doEcalDigis*process.localReco_tbsim*process.treeProducerCalibSimul)
process.p1 = cms.Path(process.doSimHits*process.doSimTB*process.doEcalDigis*process.localReco_tbsim)



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

#process.MessageLogger.debugModule = cms.untracked.vstring('g4SimHits','VtxSmeared')

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
    ,G4cout = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    ,G4cerr = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    ,BeamProfileVtxGenerator = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    )

#Add these 3 lines to put back the summary for timing information at the end of the logfile
#(needed for TimeReport report)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


#if not hasattr(process,"options") :
process.options = cms.untracked.PSet()
process.options.SkipEvent = cms.untracked.vstring('EventCorruption')


