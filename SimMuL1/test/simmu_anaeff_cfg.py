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
#w=11
#w=7

neutron=''
#neutron='n'
#neutron='hn'
#neutron='rn'

ptMethod = 33


theME42=''
#theME42='_nome42'
#theME42='_tfbx2'
#theME42='_no1'
#theME42='_mpc18'


deltaMatch=2

minTrackDR = 0.4
minTrackDR = 0.0

theMinSimTrEta = 0.8
theMinSimTrEta = -2.5
theMaxSimTrEta = 2.5

need2StWith4Hits = True
needME1With4Hits = False


suffix=''
#suffix='_rpc'
suffix='_cor2'
suffix='_pat2'

suffix='_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/NuGun_PU100_L1_pat2/'
#suffix='_pat6'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/NuGun_PU100_L1_pat6/'
suffix='_pat8'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/NuGun_PU100_L1_pat8/'

suffix='_gem_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM_NuGun_PU100_L1_pat2/'
#suffix='_gem_pat6'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM_NuGun_PU100_L1_pat6/'
#suffix='_gem_pat8'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM_NuGun_PU100_L1_pat8/'

#suffix='_gemOld_pt20_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM_NuGun_PU100_L1_pat2/'


suffix='_def_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/khotilov/GEN-SIM-NeutrinoGun_v3/GEM_NeutrinoGun_110K_pu100_DIGI_L1/2d6d66b06450f6f4eca385bc78cbc39a/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/khotilov/GEN-SIM-NeutrinoGun_v3/GEM_NeutrinoGun_pu100_DIGI_L1/2d6d66b06450f6f4eca385bc78cbc39a/']
#suffix='_gem98_pt05_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt05_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt05_pat2_128K/']
#suffix='_gem98_pt06_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt06_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt06_pat2_128K/']
#suffix='_gem98_pt10_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt10_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt10_pat2_128K/']
#suffix='_gem98_pt15_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt15_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt15_pat2_128K/']
#suffix='_gem98_pt20_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt20_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt20_pat2_128K/']
#suffix='_gem98_pt30_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt30_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt30_pat2_128K/']
#suffix='_gem98_pt40_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt40_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt40_pat2_128K/']

#suffix='_gem95_pt05_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt05_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt05_pat2_128K/']
#suffix='_gem95_pt10_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt10_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt10_pat2_128K/']
#suffix='_gem95_pt15_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt15_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt15_pat2_128K/']
#suffix='_gem95_pt20_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt20_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt20_pat2_128K/']
#suffix='_gem95_pt30_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt30_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt30_pat2_128K/']
#suffix='_gem95_pt40_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt40_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt40_pat2_128K/']


suffix='_def_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM_NuGun_PU100_L1_def_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM_NuGun_PU100_L1_def_pat8/']
#suffix='_gem98_pt05_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt05_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt05_pat8_128K/']
suffix='_gem98_pt06_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt06_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt06_pat8_128K/']
suffix='_gem98_pt10_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt10_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt10_pat8_128K/']
suffix='_gem98_pt15_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt15_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt15_pat8_128K/']
suffix='_gem98_pt20_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt20_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt20_pat8_128K/']
suffix='_gem98_pt30_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt30_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt30_pat8_128K/']
suffix='_gem98_pt40_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt40_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_NuGun_PU100_L1_pt40_pat8_128K/']

#suffix='_gem95_pt05_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt05_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt05_pat8_128K/']
#suffix='_gem95_pt10_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt10_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt10_pat8_128K/']
#suffix='_gem95_pt15_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt15_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt15_pat8_128K/']
#suffix='_gem95_pt20_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt20_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt20_pat8_128K/']
#suffix='_gem95_pt30_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt30_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt30_pat8_128K/']
#suffix='_gem95_pt40_pat8'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt40_pat8_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_NuGun_PU100_L1_pt40_pat8_128K/']


#suffix='_def_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/khotilov/MuomGUN_SIM_Pt2-50_100k/MuonGun_DIGI_L1_Pt2-50_100k/29891ddb18281fff4c42a6e5f5d4bc55/'
#suffix='_gem98_pt10_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_MuGun_PU0_L1_pt10_pat2/'
#suffix='_gem98_pt15_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_MuGun_PU0_L1_pt15_pat2/'
#suffix='_gem98_pt20_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_MuGun_PU0_L1_pt20_pat2/'
#suffix='_gem98_pt30_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_MuGun_PU0_L1_pt30_pat2/'
#suffix='_gem98_pt40_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM98_MuGun_PU0_L1_pt40_pat2/'
#suffix='_gem95_pt10_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_MuGun_PU0_L1_pt10_pat2/'
#suffix='_gem95_pt20_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_MuGun_PU0_L1_pt20_pat2/'
#suffix='_gem95_pt30_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_MuGun_PU0_L1_pt30_pat2/'
#suffix='_gem95_pt40_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM95_MuGun_PU0_L1_pt40_pat2/'

#suffix='_ntup_pt40_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT40_lpcgem/MuomGunPt40L1CSC50k_digi/82325e40d6202e6fec2dd983c477f3ca/'



#suffix='_gem_dphi_pat2'; inputDir = ['/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM_NuGun_PU100_L1_dphi_pat2_110K/','/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM_NuGun_PU100_L1_dphi_pat2_128K/']
suffix='_gem_dphi0xx_pat2'; inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/GEM_MuGun_PU0_L1_dphi_pat2/'


##### end parameters #####

thePU='000'
#thePU='050'
#thePU='100'
#thePU='200'

theProc='dimu'
#theProc='minbias'

thePre=''
#thePre='_pre2'
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



#theName = theProc+'_pu'+thePU+'_step_'+str(theStep)
#theName = theProc+'_pu'+thePU+'_redigi_'+str(theStep)
theName = theProc+'_'+neutron+'pu'+thePU+'_step_'+str(theStep)+thePre

#theFileList = 'files_'+theProc+'_3_6_2_'+neutron+'pu'+thePU+'_me42_me1a_2pi.txt'
theInputName = 'file:/tmp/out_'+theName+theME42+'.root'
#theHistoFileName = 'hf_'+theProc+'_3_6_2_'+neutron+'pu'+thePU+'_me42_me1a_2pi_step'+str(theStep)+'.root'
#theHistoFileName = 'hf_'+theProc+'_3_6_2_'+neutron+'pu'+thePU+'_me42_me1a_2pi_redigi'+str(theStep)+'.root'
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



#theHistoFileName = "h_Zmumu_pu400_2pi.root"
#theHistoFileName = "h_me42_me1a_dimu_nopu.root"
#theHistoFileName = "h_me42_me1a_dimu_pu400.root"
#theHistoFileName = "h_me42_me1a_dimu_pu400.root"

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
 

#theInputName = 'file:/uscmst1b_scratch/lpc1/lpctau/khotilov/condor/dimu_2_2_3_pu001_2pi/out_204_1500evt.root'
#theHistoFileName = 'hh_'+theProc+'_2_2_3_pu001_me42_2pi.root'
#theFileList = 'files_dimu_2_2_3_pu001_me42_2pi.txt'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(theNumberOfEvents)
)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff')
#process.load('FrontierConditions_GlobalTag_noesprefer_cff')
process.load('L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff')
#process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.Geometry.GeometryExtendedPostLS1Reco_cff')

process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
#process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
#process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
#process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
#process.load('Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometryDB_cff')
#process.load('Geometry.DTGeometryBuilder.idealForDigiDtGeometryDB_cff')
#process.load('Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff')
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
#     Rethrow = cms.untracked.vstring('ProductNotFound'),
#     FailPath = cms.untracked.vstring('ProductNotFound'),
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
#    inputCommands = cms.untracked.vstring(
#      'keep  *_*_*_*',
#      'drop *_simDtTriggerPrimitiveDigis_*_MUTRG'
#    ),
    fileNames = cms.untracked.vstring(
#    "file:/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_2_2_3/src/out_dimu.root"
#    "file:/uscmst1b_scratch/lpc1/lpctau/khotilov/condor/dimu_2_2_3/out_100_1000evt.root",
#    "dcache:/pnfs/cms/WAX/resilient/khotilov/CMSSW_2_2_3/dimu_me42_nopu_2pi/concat_20000evt.root"
#    "file:/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_2_2_3/src/out.root"
#    "file:/uscmst1b_scratch/lpc1/lpctau/khotilov/condor/dimu_2_2_3/concat_me42_v11_20000evt.root"
#    "file:/uscmst1b_scratch/lpc1/lpctau/khotilov/condor/dimu_2_2_3/concat_me42_v11_20000evt_100.root"
#      "file:/tmp/out_5k.root"
      #theInputName
      *theInputNames
      #[inputDir[16:] + x for x in ls if x.endswith('root')]
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
#SimTrackMatching.verboseSimHit = 1
#SimTrackMatching.verboseGEMDigi = 1
#SimTrackMatching.verboseCSCDigi = 1
#SimTrackMatching.verboseCSCStub = 1
#SimTrackMatching.simMuOnlyGEM = False
#SimTrackMatching.simMuOnlyCSC = False
#SimTrackMatching.discardEleHitsCSC = False
#SimTrackMatching.discardEleHitsGEM = False



process.SimMuL1StrictAll = cms.EDFilter("SimMuL1_Eff",
    doStrictSimHitToTrackMatch = cms.untracked.bool(True),
    matchAllTrigPrimitivesInChamber = cms.untracked.bool(True),
    
#    minDeltaWire = cms.untracked.int32(-2),
#    maxDeltaWire = cms.untracked.int32(0),
#    minDeltaStrip = cms.untracked.int32(1),
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

#    debugALLEVENT = cms.untracked.int32(1),
#    debugINHISTOS = cms.untracked.int32(1),
#    debugALCT = cms.untracked.int32(1),
#    debugCLCT = cms.untracked.int32(1),
#    debugLCT = cms.untracked.int32(1),
#    debugMPLCT = cms.untracked.int32(1),
#    debugTFTRACK = cms.untracked.int32(1),
#    debugTFCAND = cms.untracked.int32(1),
#    debugGMTCAND = cms.untracked.int32(1),
#    debugL1EXTRA = cms.untracked.int32(1),
##    debugRATE = cms.untracked.int32(1),

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
    
    #PTLUT = cms.PSet(
    #  LowQualityFlag = cms.untracked.uint32(4),
    #  ReadPtLUT = cms.bool(False),
    #  PtMethod = cms.untracked.uint32(ptMethod)
    #),
    SectorProcessor = process.simCsctfTrackDigis.SectorProcessor,
    
    goodChambersOnly = cms.untracked.bool(False),
    strips = process.simMuonCSCDigis.strips
)

process.SimMuL1StrictAll0 = cms.EDFilter("SimMuL1_Eff",
    doStrictSimHitToTrackMatch = cms.untracked.bool(True),
    matchAllTrigPrimitivesInChamber = cms.untracked.bool(True),
    
    minBX = cms.untracked.int32(0),
    maxBX = cms.untracked.int32(0),
    #centralBxOnlyGMT = cms.untracked.bool(False),

    minBxALCT = cms.untracked.int32(readout_windows[0][0]),
    maxBxALCT = cms.untracked.int32(readout_windows[0][1]),
    minBxCLCT = cms.untracked.int32(readout_windows[1][0]),
    maxBxCLCT = cms.untracked.int32(readout_windows[1][1]),
    minBxLCT = cms.untracked.int32(readout_windows[2][0]),
    maxBxLCT = cms.untracked.int32(readout_windows[2][1]),
    minBxMPLCT = cms.untracked.int32(readout_windows[3][0]),
    maxBxMPLCT = cms.untracked.int32(readout_windows[3][1]),
    
    minSimTrPt = cms.untracked.double(2.),
    minSimTrPhi = cms.untracked.double(-9.),
    maxSimTrPhi = cms.untracked.double(9.),
    minSimTrEta = cms.untracked.double(0.8),
    maxSimTrEta = cms.untracked.double(2.5),
    invertSimTrPhiEta = cms.untracked.bool(False),
    
    goodChambersOnly = cms.untracked.bool(False),
    strips = process.simMuonCSCDigis.strips
)

process.SimMuL1StrictAllPLUS = cms.EDFilter("SimMuL1_Eff",
    doStrictSimHitToTrackMatch = cms.untracked.bool(True),
    matchAllTrigPrimitivesInChamber = cms.untracked.bool(True),

    minBxALCT = cms.untracked.int32(readout_windows[0][0]),
    maxBxALCT = cms.untracked.int32(readout_windows[0][1]),
    minBxCLCT = cms.untracked.int32(readout_windows[1][0]),
    maxBxCLCT = cms.untracked.int32(readout_windows[1][1]),
    minBxLCT = cms.untracked.int32(readout_windows[2][0]),
    maxBxLCT = cms.untracked.int32(readout_windows[2][1]),
    minBxMPLCT = cms.untracked.int32(readout_windows[3][0]),
    maxBxMPLCT = cms.untracked.int32(readout_windows[3][1]),
    
    minSimTrPt = cms.untracked.double(2.),
    minSimTrPhi = cms.untracked.double(-9.),
    maxSimTrPhi = cms.untracked.double(9.),
    minSimTrEta = cms.untracked.double(0.8),
    maxSimTrEta = cms.untracked.double(2.5),
    invertSimTrPhiEta = cms.untracked.bool(False),

    goodChambersOnly = cms.untracked.bool(False),
    strips = process.simMuonCSCDigis.strips,

    lookAtTrackCondition = cms.untracked.int32(1)
)

process.SimMuL1StrictAllMINUS = cms.EDFilter("SimMuL1_Eff",
    doStrictSimHitToTrackMatch = cms.untracked.bool(True),
    matchAllTrigPrimitivesInChamber = cms.untracked.bool(True),

    minBxALCT = cms.untracked.int32(readout_windows[0][0]),
    maxBxALCT = cms.untracked.int32(readout_windows[0][1]),
    minBxCLCT = cms.untracked.int32(readout_windows[1][0]),
    maxBxCLCT = cms.untracked.int32(readout_windows[1][1]),
    minBxLCT = cms.untracked.int32(readout_windows[2][0]),
    maxBxLCT = cms.untracked.int32(readout_windows[2][1]),
    minBxMPLCT = cms.untracked.int32(readout_windows[3][0]),
    maxBxMPLCT = cms.untracked.int32(readout_windows[3][1]),
    
    minSimTrPt = cms.untracked.double(2.),
    minSimTrPhi = cms.untracked.double(-9.),
    maxSimTrPhi = cms.untracked.double(9.),
    minSimTrEta = cms.untracked.double(0.8),
    maxSimTrEta = cms.untracked.double(2.5),
    invertSimTrPhiEta = cms.untracked.bool(False),

    goodChambersOnly = cms.untracked.bool(False),
    strips = process.simMuonCSCDigis.strips,

    lookAtTrackCondition = cms.untracked.int32(-1)
)

process.SimMuL1Strict = cms.EDFilter("SimMuL1_Eff",
    doStrictSimHitToTrackMatch = cms.untracked.bool(True),
    matchAllTrigPrimitivesInChamber = cms.untracked.bool(False),
    
    minDeltaWire = cms.untracked.int32(0),
    maxDeltaWire = cms.untracked.int32(2),
    minDeltaStrip = cms.untracked.int32(1),
    
    doME1a = cms.untracked.bool(True),
    defaultME1a = cms.untracked.bool(defaultEmu),
    gangedME1a = cms.untracked.bool(ganged),

    minBxALCT = cms.untracked.int32(readout_windows[0][0]),
    maxBxALCT = cms.untracked.int32(readout_windows[0][1]),
    minBxCLCT = cms.untracked.int32(readout_windows[1][0]),
    maxBxCLCT = cms.untracked.int32(readout_windows[1][1]),
    minBxLCT = cms.untracked.int32(readout_windows[2][0]),
    maxBxLCT = cms.untracked.int32(readout_windows[2][1]),
    minBxMPLCT = cms.untracked.int32(readout_windows[3][0]),
    maxBxMPLCT = cms.untracked.int32(readout_windows[3][1]),
    
    minSimTrPt = cms.untracked.double(2.),
    minSimTrPhi = cms.untracked.double(-9.),
    maxSimTrPhi = cms.untracked.double(9.),
    minSimTrEta = cms.untracked.double(0.8),
    maxSimTrEta = cms.untracked.double(2.5),
    invertSimTrPhiEta = cms.untracked.bool(False),
    
    goodChambersOnly = cms.untracked.bool(False),
    strips = process.simMuonCSCDigis.strips
)

process.SimMuL1 = cms.EDFilter("SimMuL1_Eff",
    doStrictSimHitToTrackMatch = cms.untracked.bool(False),
    matchAllTrigPrimitivesInChamber = cms.untracked.bool(False),
    
    minDeltaWire = cms.untracked.int32(0),
    maxDeltaWire = cms.untracked.int32(2),
    minDeltaStrip = cms.untracked.int32(1),
    
    doME1a = cms.untracked.bool(True),
    defaultME1a = cms.untracked.bool(defaultEmu),
    gangedME1a = cms.untracked.bool(ganged),

    minBxALCT = cms.untracked.int32(readout_windows[0][0]),
    maxBxALCT = cms.untracked.int32(readout_windows[0][1]),
    minBxCLCT = cms.untracked.int32(readout_windows[1][0]),
    maxBxCLCT = cms.untracked.int32(readout_windows[1][1]),
    minBxLCT = cms.untracked.int32(readout_windows[2][0]),
    maxBxLCT = cms.untracked.int32(readout_windows[2][1]),
    minBxMPLCT = cms.untracked.int32(readout_windows[3][0]),
    maxBxMPLCT = cms.untracked.int32(readout_windows[3][1]),
    
    minSimTrPt = cms.untracked.double(2.),
    minSimTrPhi = cms.untracked.double(-9.),
    maxSimTrPhi = cms.untracked.double(9.),
    minSimTrEta = cms.untracked.double(0.8),
    maxSimTrEta = cms.untracked.double(2.5),
    invertSimTrPhiEta = cms.untracked.bool(False),
    
    goodChambersOnly = cms.untracked.bool(False),
    strips = process.simMuonCSCDigis.strips
)

#process.SimMuL1NaturalAll = cms.EDFilter("SimMuL1",
#    doStrictSimHitToTrackMatch = cms.untracked.bool(False),
#    matchAllTrigPrimitivesInChamber = cms.untracked.bool(True),
#
#    minSimTrPt = cms.untracked.double(2.),
#    minSimTrPhi = cms.untracked.double(-9.),
#    maxSimTrPhi = cms.untracked.double(9.),
#    minSimTrEta = cms.untracked.double(0.8),
#    maxSimTrEta = cms.untracked.double(2.5),
#    invertSimTrPhiEta = cms.untracked.bool(False),
#    
#    goodChambersOnly = cms.untracked.bool(False),
#    strips = process.simMuonCSCDigis.strips
#)
#

#process.output = cms.OutputModule("PoolOutputModule",
#)

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

#process.GlobalTag.globaltag = 'DESIGN60_V5::All'
process.GlobalTag.globaltag = 'POSTLS161_V12::All'

#process.GlobalTag.globaltag = 'DESIGN_37_V4::All'

#Ideal/trivial conditions - perfectly aligned and calibrated detector.
#Alignment and calibration constants = 1, with no smearing. No bad channels.


################################################################################

#Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
#from SLHCUpgradeSimulations.Configuration.postLS1Customs import digiCustoms
from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_geom_cond_digi
process=customise_csc_geom_cond_digi(process)
#call to customisation function digiCustoms imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
#process = digiCustoms(process)

from SLHCUpgradeSimulations.Configuration.muonCustoms import customize_l1extra
process=customize_l1extra(process)


##process.load("CalibMuon.Configuration.getCSCConditions_frontier_cff")
#process.cscConditions.toGet = cms.VPSet(
#        cms.PSet(record = cms.string('CSCDBGainsRcd'), tag = cms.string('CSCDBGains_ME42_offline')),
#        cms.PSet(record = cms.string('CSCDBNoiseMatrixRcd'), tag = cms.string('CSCDBNoiseMatrix_ME42_Feb2009')),
#        cms.PSet(record = cms.string('CSCDBCrosstalkRcd'), tag = cms.string('CSCDBCrosstalk_ME42_offline')),
#        cms.PSet(record = cms.string('CSCDBPedestalsRcd'), tag = cms.string('CSCDBPedestals_ME42_offline'))
#)
#process.es_prefer_cscConditions = cms.ESPrefer("PoolDBESSource","cscConditions")
##process.es_prefer_cscBadChambers = cms.ESPrefer("PoolDBESSource","cscBadChambers")
process.CSCGeometryESModule.applyAlignment = False



#process.Timing = cms.Service("Timing")
#process.Tracer = cms.Service("Tracer")


#process.ana_seq = cms.Sequence(process.SimMuL1StrictAll+process.SimMuL1NaturalAll)
#process.ana_seq = cms.Sequence(process.SimMuL1StrictAll+process.SimMuL1StrictAll0+process.SimMuL1StrictAll3)
#process.ana_seq = cms.Sequence(process.SimMuL1StrictAll+process.SimMuL1StrictAll0)
#process.ana_seq = cms.Sequence(process.SimMuL1StrictAll+process.SimMuL1StrictAll0+process.SimMuL1StrictAllPLUS+process.SimMuL1StrictAllMINUS)
#process.ana_seq = cms.Sequence(process.SimMuL1StrictAll+process.SimMuL1StrictAllPLUS+process.SimMuL1StrictAllMINUS+process.SimMuL1Strict+process.SimMuL1)
#process.ana_seq = cms.Sequence(process.SimMuL1StrictAll+process.SimMuL1Strict+process.SimMuL1)
process.ana_seq = cms.Sequence(process.SimMuL1StrictAll)
process.reader_seq = cms.Sequence(process.lctreader)
process.cscdigi_seq = cms.Sequence(process.cscdigiana)


process.l1extra_step        = cms.Path(process.L1Extra)
process.ana_step        = cms.Path(process.ana_seq)
process.reader_step     = cms.Path(process.reader_seq)
process.cscdigi_step    = cms.Path(process.cscdigi_seq)

process.schedule = cms.Schedule(
#    process.l1extra_step,
#    process.cscdigi_step,
    process.ana_step
#    process.reader_step
)
