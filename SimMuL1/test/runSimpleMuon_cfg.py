import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("SIMMUON")


## CMSSW RELEASE
cmssw = os.getenv( "CMSSW_VERSION" )

## steering
events = 10000
defaultEmu = False
ganged = False
pileup='000'
sample='dimu'
globalTag = 'upgrade2019'
#sample='minbias'

## input
from GEMCode.SimMuL1.GEMCSCTriggerSamplesLib import files
suffix = '_gem98_pt2-50_PU0_pt20_new'
inputDir = files[suffix]
theInputFiles = []
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
    
##inputFiles = ['file:out_SingleMuPt10Fwd_GEM2019_8PartIncRad_DIGI_L1.root']
#print "InputFiles: ", theInputFiles

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
outputFileName = 'hp_SimpleMuon_' + sample + "_" + cmssw + "_" + globalTag + "_pu%s"%(pileup) + '_w%d'%(w) + suffix + '_eff.test.root'
#outputFileName = 'gem_trigger_eff_ana.root'
print "outputFile:", outputFileName

process.load("FWCore.MessageService.MessageLogger_cfi")
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
process.load('L1Trigger.CSCTrackFinder.csctfTrackDigisUngangedME1a_cfi')
process.simCsctfTrackDigis = process.csctfTrackDigisUngangedME1a.clone()
process.simCsctfTrackDigis.DTproducer = cms.untracked.InputTag("simDtTriggerPrimitiveDigis")
process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("simCscTriggerPrimitiveDigis","MPCSORTED")
process.simCsctfTrackDigis.SectorProcessor.isCoreVerbose = cms.bool(True)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(events) )

#inputFile = ['file:/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_SLHC1/src/out_L1_MuonGun_neweta_PU100_Pt20_50k_digi_preTrig2.root']
#inputFile = ['file:/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_SLHC1/src/out_sim_singleMuPt100Fwdv2.root']
#inputFile = ['file:/afs/cern.ch/user/d/dildick/out_digi_1_1_QXc.root']
#inputFile = ['file:/afs/cern.ch/user/d/dildick/out_digi_30_1_mZD.root']

#inputFile = ['file:/afs/cern.ch/user/d/dildick/work/GEM/digiFiles/GEM_NeutrinoGun_pu100_DIGI_L1/out_21_1_NlZ.root']
#inputFile = ['file:../../../../digiFiles/MuonGun_neweta_PU100_Pt20_50k_digi/Muon_DIGI_1632_1_baH.root']

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        #'file:myfile.root'
        *theInputFiles
    )
)

process.load('GEMCode.SimMuL1.SimpleMuon_cfi')
process.SimpleMuon.strips = process.simMuonCSCDigis.strips
readout_windows = [ [5,7],[5,7],[5,7],[5,7] ]
"""
process.SimpleMuon.minBxALCT = readout_windows[0][0]
process.SimpleMuon.maxBxALCT = readout_windows[0][1]
process.SimpleMuon.minBxCLCT = readout_windows[1][0]
process.SimpleMuon.maxBxCLCT = readout_windows[1][1]
process.SimpleMuon.minBxLCT = readout_windows[2][0]
process.SimpleMuon.maxBxLCT = readout_windows[2][1]
process.SimpleMuon.minBxMPLCT = readout_windows[3][0]
process.SimpleMuon.maxBxMPLCT = readout_windows[3][1]
"""

#outputFileName = 'output.test.root'
process.TFileService = cms.Service("TFileService",
    fileName = cms.string(outputFileName)
)

process.p = cms.Path(process.SimpleMuon)
