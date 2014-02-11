import FWCore.ParameterSet.Config as cms
import sys, os

# Hack to add "test" directory to the python path.
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'], 'src/L1Trigger/CSCTriggerPrimitives/test'))
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'], 'src'))

## initialization
process = cms.Process('GEMCSCTRGANA')

## CMSSW RELEASE
cmssw = os.getenv( "CMSSW_VERSION" )

## steering
events = 20
defaultEmu = False
pileup='000'
sample='dimu'
globalTag = 'upgrade2019'
#sample='minbias'


## input
from GEMCode.SimMuL1.GEMCSCTriggerSamplesLib import files
suffix = '_gem98_pt2-50_PU0_pt0_new'
inputDir = files[suffix]
theInputFiles = []
import os
for d in range(len(inputDir)):
  my_dir = inputDir[d]
  if not os.path.isdir(my_dir):
    print "ERROR: This is not a valid directory: ", my_dir
    if d==len(inputDir)-1:
      print "ERROR: No input files were selected"
      exit()
    continue
  print "Proceed to next directory"
  ls = os.listdir(my_dir)
  ## this works only if you pass the location on pnfs - FIXME for files staring with store/user/... 
  theInputFiles.extend([my_dir[16:] + x for x in ls if x.endswith('root')])
    
print "InputFiles: ", theInputFiles

## readout windows
w=3
if w==3:
    readout_windows = [ [5,7],[5,7],[5,7],[5,7] ]
if w==11:
    readout_windows = [ [1,11],[1,11],[1,11],[1,11] ]
if w==7:
    readout_windows = [ [5,11],[5,11],[5,11],[5,11] ]
if w==61:
    readout_windows = [ [5,10],[1,11],[1,11],[1,11] ]
 
## output
outputFileName = 'hp_' + sample + "_" + cmssw + "_" + globalTag + "_pu%s"%(pileup) + '_w%d'%(w) + suffix + '_eff.root'
print "outputFile:", outputFileName

# import of standard configurations
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2019_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('Configuration.StandardSequences.Digi_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load("Configuration.StandardSequences.L1Emulator_cff")
process.load("Configuration.StandardSequences.L1Extra_cff")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")

## GEM geometry customization
from Geometry.GEMGeometry.gemGeometryCustoms import custom_GE11_6partitions_v1
process = custom_GE11_6partitions_v1(process)

## upgrade CSC TrackFinder                                                                                                                                               
from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_L1TrackFinder
process = customise_csc_L1TrackFinder(process)
process.simCsctfTrackDigis.SectorProcessor.isCoreVerbose = cms.bool(True)

process.load('CSCTriggerPrimitivesReader_cfi')
process.lctreader.debug = False
process.lctreader.dataLctsIn = False
process.lctreader.CSCLCTProducerEmul = "simCscTriggerPrimitiveDigis"
process.lctreader.CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi")
process.lctreader.CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")

process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(False),
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    fileNames = cms.untracked.vstring(
      *theInputFiles
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(events)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string(outputFileName)
)

process.load('GEMCode.SimMuL1.GEMCSCTriggerEfficiency_cfi')
process.GEMCSCTriggerEfficiency.sectorProcessor = process.simCsctfTrackDigis.SectorProcessor
process.GEMCSCTriggerEfficiency.strips = process.simMuonCSCDigis.strips
process.GEMCSCTriggerEfficiency.minBxALCT = readout_windows[0][0]
process.GEMCSCTriggerEfficiency.maxBxALCT = readout_windows[0][1]
process.GEMCSCTriggerEfficiency.minBxCLCT = readout_windows[1][0]
process.GEMCSCTriggerEfficiency.maxBxCLCT = readout_windows[1][1]
process.GEMCSCTriggerEfficiency.minBxLCT = readout_windows[2][0]
process.GEMCSCTriggerEfficiency.maxBxLCT = readout_windows[2][1]
process.GEMCSCTriggerEfficiency.minBxMPLCT = readout_windows[3][0]
process.GEMCSCTriggerEfficiency.maxBxMPLCT = readout_windows[3][1]
process.GEMCSCTriggerEfficiency.minNHitsChamber = cms.untracked.int32(4)
process.GEMCSCTriggerEfficiency.requireME1WithMinNHitsChambers = cms.untracked.bool(True)
process.GEMCSCTriggerEfficiency.minSimTrPt = cms.untracked.double(2)
GEMmatching = process.GEMCSCTriggerEfficiency.simTrackMatching
GEMmatching.gemRecHit.input = ""
#SimTrackMatching.verboseSimHit = 1
#SimTrackMatching.verboseGEMDigi = 1
#SimTrackMatching.verboseCSCDigi = 1
#SimTrackMatching.verboseCSCStub = 1
#SimTrackMatching.simMuOnlyGEM = False
#SimTrackMatching.simMuOnlyCSC = False
#SimTrackMatching.discardEleHitsCSC = False
#SimTrackMatching.discardEleHitsGEM = False

## SLHC customization
from SLHCUpgradeSimulations.Configuration.muonCustoms import *
process = unganged_me1a_geometry(process)
process = customise_csc_L1Extra_allsim(process)

## sequence, path and schedule
process.ana_seq = cms.Sequence(process.GEMCSCTriggerEfficiency)
process.reader_seq = cms.Sequence(process.lctreader)

process.l1extra_step = cms.Path(process.L1Extra)
process.ana_step     = cms.Path(process.ana_seq)
process.reader_step  = cms.Path(process.reader_seq)

process.schedule = cms.Schedule(
#    process.l1extra_step,
    process.ana_step
#    process.reader_step
)

