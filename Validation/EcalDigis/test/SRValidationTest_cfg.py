# The following comments couldn't be translated into the new config version:

#,
import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSelectiveReadoutValid")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# geometry (Only Ecal)
#process.load("Geometry.EcalCommonData.EcalOnly_cfi")
# geometry (Only Ecal)
process.load("Geometry.EcalCommonData.EcalOnly_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")


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
process.ecalSelectiveReadoutValidation.outputFile = 'srvalid_hists.root'

#ECAL reco sequence:
process.load("RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff")
process.ecalWeightUncalibRecHit.EBdigiCollection = cms.InputTag("simEcalDigis", "ebDigis")
process.ecalWeightUncalibRecHit.EEdigiCollection = cms.InputTag("simEcalDigis", "eeDigis")

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

#Pythia configuration to generate multijet event with pt_hat between 30
#and 50 GeV/c
process.load("Configuration.Generator.QCD_Pt_30_50_cfi")

process.tpparams12 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPhysicsConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.DQM.collectorHost = ''

process.g4SimHits.Generator.HepMCProductLabel = 'source'

process.simEcalDigis.writeSrFlags = True

# detector response simulation path:
process.detSim = cms.Sequence(process.VtxSmeared*process.g4SimHits)

# processing path:
process.p1 = cms.Path(process.detSim*process.mix*process.simEcalUnsuppressedDigis*process.simEcalTriggerPrimitiveDigis*process.simEcalDigis*process.ecalWeightUncalibRecHit*process.ecalRecHit*process.ecalSelectiveReadoutValidation)

process.outpath = cms.EndPath(process.o1)


process.simEcalDigis.srpEndcapLowInterestChannelZS = -1.e9 #-0.06
process.simEcalDigis.srpBarrelLowInterestChannelZS = -1.e9 #-0.035

