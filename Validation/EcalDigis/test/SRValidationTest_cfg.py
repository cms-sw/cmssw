# The following comments couldn't be translated into the new config version:

#,
import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSelectiveReadoutValid")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# geometry (Only Ecal)
process.load("Geometry.EcalCommonData.EcalOnly_cfi")

# DQM services:
process.load("DQMServices.Core.DQM_cfg")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# event vertex smearing - applies only once (internal check)
# Note : all internal generators will always do (0,0,0) vertex
#
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

# run simulation, with EcalHits Validation specific watcher 
process.load("SimG4Core.Application.g4SimHits_cfi")

#  replace g4SimHits.Watchers = {
#       { string type = "EcalSimHitsValidProducer"
#         untracked string instanceLabel="EcalValidInfo"
#         untracked bool verbose = false
#       }
#  }
# Mixing Module
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

# ECAL digitization sequence
process.load("SimCalorimetry.Configuration.ecalDigiSequence_cff")

# ECAL digis validation sequence
#include "Validation/EcalDigis/data/ecalDigisValidationSequence.cff"
# Defines Ecal seletive readout validation module, ecalSelectiveReadoutValidation:
process.load("Validation.EcalDigis.ecalSelectiveReadoutValidation_cfi")

#ECAL reco sequence:
process.load("RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('QCD_pt30_50_all_SRValidation.root')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        simEcalUnsuppressedDigis = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.source = cms.Source("PythiaSource",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring('MSEL=1               ! QCD hight pT processes', 
            'CKIN(3)=30.          ! minimum pt hat for hard interactions', 
            'CKIN(4)=50.          ! maximum pt hat for hard interactions'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters'),
        pythiaUESettingsBlock = cms.PSet(
            pythiaUESettings = cms.vstring('MSTJ(11)=3     ! Choice of the fragmentation function', 
                'MSTJ(22)=2     ! Decay those unstable particles', 
                'PARJ(71)=10 .  ! for which ctau  10 mm', 
                'MSTP(2)=1      ! which order running alphaS', 
                'MSTP(33)=0     ! no K factors in hard cross sections', 
                'MSTP(51)=7     ! structure function chosen', 
                'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default', 
                'MSTP(82)=4     ! Defines the multi-parton model', 
                'MSTU(21)=1     ! Check on possible errors during program execution', 
                'PARP(82)=1.9409   ! pt cutoff for multiparton interactions', 
                'PARP(89)=1960. ! sqrts for which PARP82 is set', 
                'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 
                'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 
                'PARP(90)=0.16  ! Multiple interactions: rescaling power', 
                'PARP(67)=2.5    ! amount of initial-state radiation', 
                'PARP(85)=1.0  ! gluon prod. mechanism in MI', 
                'PARP(86)=1.0  ! gluon prod. mechanism in MI', 
                'PARP(62)=1.25   ! ', 
                'PARP(64)=0.2    ! ', 
                'MSTP(91)=1     !', 
                'PARP(91)=2.1   ! kt distribution', 
                'PARP(93)=15.0  ! ')
        )
    )
)

process.tpparams12 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPhysicsConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.detSim = cms.Sequence(process.VtxSmeared*process.g4SimHits)
process.p1 = cms.Path(process.detSim*process.mix*process.simEcalUnsuppressedDigis*process.simEcalTriggerPrimitiveDigis*process.simEcalDigis*process.ecalWeightUncalibRecHit*process.ecalRecHit*process.ecalSelectiveReadoutValidation)
process.outpath = cms.EndPath(process.o1)
process.DQM.collectorHost = ''
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.simEcalDigis.writeSrFlags = True

