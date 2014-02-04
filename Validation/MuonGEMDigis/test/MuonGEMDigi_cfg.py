import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")


process.load("Configuration.EventContent.EventContent_cff")
process.load("DQMServices.Core.DQM_cfg")
## Standard sequence
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2019_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load("FWCore.MessageLogger.MessageLogger_cfi")

## TrackingComponentsRecord required for matchers
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')

## global tag for 2019 upgrade studies
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

from DQMServices.Components.DQMEnvironment_cfi import *

from DQMServices.Examples.test.ConverterTester_cfi import *
#from DQMServices.Components.MEtoEDMConverter.cfi import *

DQMStore = cms.Service("DQMStore")

dqmSaver.convention = 'Offline'

process.MEtoEDMConverter.Verbosity= cms.untracked.int32(1)
process.MEtoEDMConverter.Frequency= cms.untracked.int32(1)
process.MEtoEDMConverter.deleteAfterCopy = cms.untracked.bool(False)



process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
#        'file:/pnfs/user/geonmo/CMSSW_6_2_0_SLHC1/src/Validation/MuonGEMDigis/data/out_digi_2.root'
         'file:/pnfs/user/unclok/digi/4954.uosaf0008.sscc.uos.ac.kr/out_sim.root'   
 )
)

process.o1 = cms.OutputModule("PoolOutputModule",
	outputCommands = cms.untracked.vstring('keep *'),
	fileName = cms.untracked.string('out_digi_validation.root')
)

from Validation.MuonGEMDigis.simTrackMatching_cfi import SimTrackMatching
process.load('Validation.MuonGEMDigis.MuonGEMDigis_cfi') 
process.gemDigiValidation.outputFile= cms.string('valid.root')
process.gemDigiValidation.simTrackMatching = SimTrackMatching
       	


process.p = cms.Path(process.gemDigiValidation*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.o1)
