# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt100_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW,RAW2DIGI,L1Reco,RECO --conditions auto:run2_mc --magField 38T_PostLS1 --datatier GEN-SIM --geometry GEMCosmicStand --eventcontent FEVTDEBUGHLT --era phase2_muon -n 100 --fileout out_reco.root
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.phase2_muon)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Geometry.GEMGeometry.GeometryGEMCosmicStand_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('EventFilter.GEMRawToDigi.muonGEMDigis_cfi')
process.load('RecoLocalMuon.GEMRecHit.gemLocalReco_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:out_digi.root'),
)
process.options = cms.untracked.PSet()

# Output definition
process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(10485760),
    fileName = cms.untracked.string('out_reco.root'),
    outputCommands = cms.untracked.vstring( ('keep *')),
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:dqm.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

#process.FEVTDEBUGHLToutput.outputCommands.append()

# Additional output definition
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.load('RecoMuon.TrackingTools.MuonServiceProxy_cff')
process.MuonServiceProxy.ServiceParameters.Propagators.append('StraightLinePropagator')

process.GEMCosmicMuon = cms.EDProducer("GEMCosmicMuon",
                                       process.MuonServiceProxy,
                                       gemRecHitLabel = cms.InputTag("gemRecHits"),
                                       doInnerSeeding = cms.bool(False),
                                       trackChi2 = cms.double(10000.0),
                                       trackResX = cms.double(5.0),
                                       trackResY = cms.double(15.0),
                                       MuonSmootherParameters = cms.PSet(
                                           PropagatorAlong = cms.string('SteppingHelixPropagatorAny'),
                                           PropagatorOpposite = cms.string('SteppingHelixPropagatorAny'),
                                           RescalingFactor = cms.double(5.0)
                                           ),
                                       )
process.GEMCosmicMuon.ServiceParameters.GEMLayers = cms.untracked.bool(True)
process.GEMCosmicMuon.ServiceParameters.CSCLayers = cms.untracked.bool(False)
process.GEMCosmicMuon.ServiceParameters.RPCLayers = cms.bool(False)
process.GEMCosmicMuonInSide = process.GEMCosmicMuon.clone(doInnerSeeding = cms.bool(True))

process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(1)
)
process.load('Validation.GEMCosmicMuonStand.GEMCosmicMuonStandEfficiency_cff')
process.load('Validation.GEMCosmicMuonStand.GEMCosmicMuonStandSim_cff')

# Path and EndPath definitions
process.reconstruction_step = cms.Path(process.GEMCosmicMuon+process.GEMCosmicMuonInSide)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.validation_step = cms.Path(process.gemCosmicMuonStandSim+process.gemCosmicMuonStandEfficiency)
process.dqmoffline_step = cms.EndPath(process.DQMOffline)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
# Schedule definition
process.schedule = cms.Schedule(process.reconstruction_step,
                                process.validation_step,
                                process.DQMoutput_step,
                                process.FEVTDEBUGHLToutput_step)

