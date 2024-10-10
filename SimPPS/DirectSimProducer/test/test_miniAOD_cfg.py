import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from Configuration.Eras.Modifier_ctpps_directSim_cff import ctpps_directSim
process = cms.Process('PPS', ctpps_directSim, eras.Run3_2023)

detailedLog=True
Year="2023"
Period="PreTS1B"
Profile="profile_"+Year+"_"+Period
print(Profile)

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SimPPS.Configuration.directSimPPS_cff')
process.load('RecoPPS.Configuration.recoCTPPS_cff')

# minimum of logs
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.threshold = cms.untracked.string('')
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

if detailedLog:
  process.MessageLogger = cms.Service("MessageLogger",
      destinations = cms.untracked.vstring('detailedInfo','cerr'),
      detailedInfo = cms.untracked.PSet(
          threshold = cms.untracked.string('INFO') 
      ),
      cerr = cms.untracked.PSet(
          threshold  = cms.untracked.string('WARNING') 
      )
  )

# period
process.ctppsCompositeESSource.periods = [process.profile_2023_PreTS1B]

# global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '140X_mcRun3_2024_realistic_v20', '')

# raw data source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        "/store/mc/Run3Summer23DRPremix/GGToMuMu_PT-25_El-El_13p6TeV_lpair/AODSIM/130X_mcRun3_2023_realistic_v14-v2/2560000/94caefe9-1efd-4c9e-99ca-9861a85c11ea.root"
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    beamDivergenceVtxGenerator = cms.PSet(initialSeed = cms.untracked.uint32(3849)),
    ppsDirectProtonSimulation = cms.PSet(initialSeed = cms.untracked.uint32(4981))
)

from SimPPS.DirectSimProducer.matching_cff import matchDirectSimOutputsAOD
matchDirectSimOutputsAOD(process)

if detailedLog:
  print('Verbosity of DirectSimProducer = '+str(process.ppsDirectProtonSimulation.verbosity))
  process.ppsDirectProtonSimulation.verbosity = cms.untracked.uint32(1)
  print('beamEnergy from profile = '+str(process.profile_2023_PreTS1B.ctppsLHCInfo.beamEnergy))

process.p = cms.Path(
    process.directSimPPS
    * process.recoDirectSimPPS
)

# output configuration
from RecoPPS.Configuration.RecoCTPPS_EventContent_cff import RecoCTPPSAOD
RecoCTPPSAOD.outputCommands.extend(cms.untracked.vstring(
    'keep *_genPUProtons_*_*',
    'keep *_genParticles_*_*'
    )
)

process.output = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('file:output'+'_'+Year+'_'+Period+'.root'),
    outputCommands = RecoCTPPSAOD.outputCommands
)

process.outpath = cms.EndPath(process.output)
