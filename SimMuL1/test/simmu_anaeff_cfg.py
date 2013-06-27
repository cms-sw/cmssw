import FWCore.ParameterSet.Config as cms

# Hack to add "test" directory to the python path.
import sys, os
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'], 'src/L1Trigger/CSCTriggerPrimitives/test'))
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'], 'src'))

process = cms.Process('simmul')

##### parameters #####

theNumberOfEvents = 1000000

defaultEmu = False
ganged = True
ganged = False

w=61
w=3
neutron=''
ptMethod = 33
theME42=''
deltaMatch=2
minTrackDR = 0.4
minTrackDR = 0.0
theMinSimTrEta = 0.8
theMinSimTrEta = -2.5
theMaxSimTrEta = 2.5
need2StWith4Hits = True
needME1With4Hits = False


suffix=''
suffix='_gem_dphi0xx_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM_MuGun_PU0_L1_dphi_pat2/'


##### end parameters #####

thePU='000'
theProc='dimu'
thePre=''
thePre='_pre3'

theStep = 50

'''
thePU = os.getenv("thePU")
theProc = os.getenv("theProc")
theStep = os.getenv("theStep")

if thePU is None: thePU = '000'
if theProc is None: theProc = 'dimu'
if theStep is None: theStep = 50
'''
theStep = int(theStep)

print "proc=%s PU=%s step=%d"%(theProc,thePU,theStep)

if (theStep<=5): defaultEmu = True


if theProc=='minbias':
  minTrackDR = 0.
  need2StWith4Hits = False
  needME1With4Hits = False

if theStep==3:
  theMinSimTrEta = 0.8
  #theMinSimTrEta = -2.1
  theMaxSimTrEta = 2.1


theName = theProc+'_'+neutron+'pu'+thePU+'_step_'+str(theStep)+thePre
theInputName = 'file:/tmp/out_'+theName+theME42+'.root'
theFileList = 'files_'+theProc+'_3_6_2_'+neutron+'pu'+thePU+'_me42_me1a_2pi'+thePre+'.txt'
theHistoFileName = 'hp_'+theProc+'_6_0_1_POSTLS161_V12_'+'_'+neutron+'pu'+thePU+'_w'+str(w)+theME42+suffix+'.root'

if ganged:
  theName = theProc+'_'+neutron+'pu'+thePU+'_g_step_'+str(theStep)+thePre+''
  theInputName = 'file:/tmp/out_'+theName+theME42+'.root'
  theHistoFileName = 'hp_'+theProc+'_6_0_1_POSTLS161_V12_'+'_'+neutron+'pu'+thePU+'_w'+str(w)+theME42+suffix+'.root'

'''
theInputName = 'file:out_dimu_pu000_step_40_pre3_fx0.root'
theInputName = 'file:out_dimu_emu_fix0.root'

theInputNames = [
  '/store/relval/CMSSW_6_0_1_PostLS1v1-POSTLS161_V10/RelValSingleMuPt100_UPGpostls1/GEN-SIM-DIGI-RAW/v1/00000/40B08FFE-FE25-E211-B89A-0018F3D096D4.root',
  '/store/relval/CMSSW_6_0_1_PostLS1v1-POSTLS161_V10/RelValSingleMuPt100_UPGpostls1/GEN-SIM-DIGI-RAW/v1/00000/28AFFD27-FE25-E211-A6D9-0026189438BF.root']

theInputNames = [
  'file:step2.root','file:step2_2.root'
]

theInputNames = [
  '/store/relval/CMSSW_6_0_1_PostLS1v2-POSTLS161_V12/RelValSingleMuPt10_UPGpostls1/GEN-SIM-DIGI-RAW/v1/00000/C636EB07-DE2F-E211-A741-00261894398B.root']
'''

print theHistoFileName
#print theInputName
print inputDir


readout_windows = [ [5,7],[5,7],[5,7],[5,7] ]
if w==3:
    readout_windows = [ [5,7],[5,7],[5,7],[5,7] ]
if w==11:
    readout_windows = [ [1,11],[1,11],[1,11],[1,11] ]
if w==7:
    readout_windows = [ [5,11],[5,11],[5,11],[5,11] ]
if w==61:
    readout_windows = [ [5,10],[1,11],[1,11],[1,11] ]
 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(theNumberOfEvents)
)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('Configuration.Geometry.GeometryExtendedPostLS1Reco_cff')
process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
process.load('Geometry.GEMGeometry.gemGeometry_cfi')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load("Configuration.StandardSequences.L1Emulator_cff")
process.load("Configuration.StandardSequences.L1Extra_cff")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load('Configuration.StandardSequences.MagneticField_cff')
# Parametrized magnetic field (new mapping)
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")


process.load('L1Trigger.CSCTrackFinder.csctfTrackDigisUngangedME1a_cfi')
process.simCsctfTrackDigis = process.csctfTrackDigisUngangedME1a
process.simCsctfTrackDigis.DTproducer = cms.untracked.InputTag("simDtTriggerPrimitiveDigis")
process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("simCscTriggerPrimitiveDigis","MPCSORTED")
process.simCsctfTrackDigis.SectorProcessor.isCoreVerbose = cms.bool(True)

################################################################################

process.load("CSCTriggerPrimitivesReader_cfi")

process.lctreader.debug = False
process.lctreader.dataLctsIn = False
process.lctreader.CSCLCTProducerEmul = "simCscTriggerPrimitiveDigis"
process.lctreader.CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi")
process.lctreader.CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")

################################################################################

process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(False),
    wantSummary = cms.untracked.bool(True)
)

if type(inputDir) != list: 
    inputDir = [inputDir]

theInputNames = []
import os
for d in inputDir:
  ls = os.listdir(d)
  theInputNames.extend([d[16:] + x for x in ls if x.endswith('root')])

process.source = cms.Source("PoolSource",
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    fileNames = cms.untracked.vstring(
      *theInputNames
    )
)

'''
ff = open(theFileList, "r")
files = ff.read().split('\n')
ff.close()

vstri=[]
for i in range (0,len(files)):
    if len(files[i])==0: continue
    if files[i].find('pnfs') > -1:
	vstri.append('dcap:'+ files[i])
    else:
	vstri.append('file:'+ files[i])

process.source.fileNames = vstri
'''


process.TFileService = cms.Service("TFileService",
    fileName = cms.string(theHistoFileName)
)


from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching


process.SimMuL1StrictAll = cms.EDFilter("SimMuL1",
    doStrictSimHitToTrackMatch = cms.untracked.bool(True),
    matchAllTrigPrimitivesInChamber = cms.untracked.bool(True),
    
    minDeltaWire = cms.untracked.int32(-1*deltaMatch),
    maxDeltaWire = cms.untracked.int32(deltaMatch),
    minDeltaStrip = cms.untracked.int32(deltaMatch),
    
    minNStWith4Hits = cms.untracked.int32(0),
    requireME1With4Hits = cms.untracked.bool(False),
    
    simTrackGEMMatching = SimTrackMatching.clone(),
    gemPTs = cms.vdouble(0., 6., 10., 15., 20., 30., 40.),
    gemDPhisOdd = cms.vdouble(1., 0.0182579,   0.01066 , 0.00722795 , 0.00562598 , 0.00416544 , 0.00342827),
    gemDPhisEven = cms.vdouble(1., 0.00790009, 0.00483286, 0.0036323, 0.00304879, 0.00253782, 0.00230833),

    doME1a = cms.untracked.bool(True),
    defaultME1a = cms.untracked.bool(defaultEmu),
    gangedME1a = cms.untracked.bool(ganged),
    
    lightRun = cms.untracked.bool(False),

    minBxALCT = cms.untracked.int32(readout_windows[0][0]),
    maxBxALCT = cms.untracked.int32(readout_windows[0][1]),
    minBxCLCT = cms.untracked.int32(readout_windows[1][0]),
    maxBxCLCT = cms.untracked.int32(readout_windows[1][1]),
    minBxLCT = cms.untracked.int32(readout_windows[2][0]),
    maxBxLCT = cms.untracked.int32(readout_windows[2][1]),
    minBxMPLCT = cms.untracked.int32(readout_windows[3][0]),
    maxBxMPLCT = cms.untracked.int32(readout_windows[3][1]),

    minSimTrackDR = cms.untracked.double(minTrackDR),

    minSimTrPt = cms.untracked.double(2.),
    minSimTrPhi = cms.untracked.double(-9.),
    maxSimTrPhi = cms.untracked.double(9.),
    #minSimTrEta = cms.untracked.double(0.8),
    minSimTrEta = cms.untracked.double(theMinSimTrEta),
    maxSimTrEta = cms.untracked.double(theMaxSimTrEta),
    invertSimTrPhiEta = cms.untracked.bool(False),
    
    SectorProcessor = process.simCsctfTrackDigis.SectorProcessor,
    
    goodChambersOnly = cms.untracked.bool(False),
    strips = process.simMuonCSCDigis.strips
)

process.cscdigiana = cms.EDAnalyzer('CSCDigiAnalizer',
    fillTree = cms.untracked.bool(False)
)

################################################################################
# load configuration for ME42 and *ganged* ME1A

#process.load('FullSim_Configure_ME42_ME1A_cff')

if ganged:
  process.CSCGeometryESModule.useGangedStripsInME1a = True
  process.idealForDigiCSCGeometry.useGangedStripsInME1a = True

################################################################################
# Global conditions tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions#Global_Tags_for_Monte_Carlo_Prod

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = 'POSTLS161_V12::All'

#Ideal/trivial conditions - perfectly aligned and calibrated detector.
#Alignment and calibration constants = 1, with no smearing. No bad channels.


################################################################################

#Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
#from SLHCUpgradeSimulations.Configuration.postLS1Customs import digiCustoms
from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_geom_cond_digi
process=customise_csc_geom_cond_digi(process)

from SLHCUpgradeSimulations.Configuration.muonCustoms import customize_l1extra
process=customize_l1extra(process)


process.CSCGeometryESModule.applyAlignment = False


process.ana_seq = cms.Sequence(process.SimMuL1StrictAll)

process.l1extra_step        = cms.Path(process.L1Extra)
process.ana_step        = cms.Path(process.ana_seq)


process.schedule = cms.Schedule(
  process.ana_step
)
