## This configuration runs the DIGI+L1Emulator step
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("MUTRG")

## Standard sequence
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2019_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load("Configuration.StandardSequences.L1Emulator_cff")
process.load("Configuration.StandardSequences.L1Extra_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring("log", "debug", "errors"),
    statistics = cms.untracked.vstring("stat"),
    # No constraint on log.txt content...
    log = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        lineLength = cms.untracked.int32(132),
        noLineBreaks = cms.untracked.bool(True)
    ),
    debug = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        threshold = cms.untracked.string("DEBUG"),
        lineLength = cms.untracked.int32(132),
        noLineBreaks = cms.untracked.bool(True)
    ),
    errors = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        threshold = cms.untracked.string("ERROR")
    ),
    stat = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        threshold = cms.untracked.string("INFO")
    ),
    # turn on the following to get LogDebug output
    # ============================================
    # debugModules = cms.untracked.vstring("*"),
    debugModules = cms.untracked.vstring("simCscTriggerPrimitiveDigis")

    )

################### Take inputs from crab.cfg file ##############
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.register ('pu',
                  0,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.float,
                  "PU: 100  default")

options.register ('ptdphi',
                  "pt05",
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.string,
                  "ptdphi: 5 GeV/c default")

import sys
print sys.argv

if len(sys.argv) > 0:
    last = sys.argv.pop()
    sys.argv.extend(last.split(","))
    print sys.argv
    
if hasattr(sys, "argv") == True:
    options.parseArguments()
    pu = options.pu
    ptdphi = options.ptdphi
    print 'Using pu: %f' % pu
    print 'Using ptdphi: %s GeV' % ptdphi
    
#--------------------------------------------------------------------------------


## global tag for 2019 upgrade studies
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

#process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# GEM digitizer
process.load('SimMuon.GEMDigitizer.muonGEMDigis_cfi')
# GEM-CSC trigger pad digi producer
process.load('SimMuon.GEMDigitizer.muonGEMCSCPadDigis_cfi')
# customization of the process.pdigi sequence to add the GEM digitizer
from SimMuon.GEMDigitizer.customizeGEMDigi import *
#process = customize_digi_addGEM(process)  # run all detectors digi
process = customize_digi_addGEM_muon_only(process) # only muon+GEM digi
#process = customize_digi_addGEM_gem_only(process)  # only GEM digi


## GEM geometry customization
use6part = True
if use6part:
    mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v4/gemf.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v4/gemf.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v2/gemf.xml')

## upgrade CSC geometry customizations
from SLHCUpgradeSimulations.Configuration.muonCustoms import unganged_me1a_geometry, digitizer_timing_pre3_median
process = unganged_me1a_geometry(process)
process = digitizer_timing_pre3_median(process)

## upgrade CSC L1 customizations: GEM-CSC emulator
process.load('L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigisPostLS1_cfi')
process.simCscTriggerPrimitiveDigis = process.cscTriggerPrimitiveDigisPostLS1.clone()
process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag('simMuonCSCDigis', 'MuonCSCComparatorDigi')
process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag('simMuonCSCDigis', 'MuonCSCWireDigi')

## GEM-CSC bending angle library
process.simCscTriggerPrimitiveDigis.gemPadProducer =  cms.untracked.InputTag("simMuonGEMCSCPadDigis","")
process.simCscTriggerPrimitiveDigis.clctSLHC.clctPidThreshPretrig = 2
process.simCscTriggerPrimitiveDigis.clctSLHC.clctNplanesHitPretrig = 3
process.simCscTriggerPrimitiveDigis.clctSLHC.clctNplanesHitPattern = 3
#process.simCscTriggerPrimitiveDigis.clctParam07.clctPidThreshPretrig = 2
tmb = process.simCscTriggerPrimitiveDigis.tmbSLHC
tmb.gemMatchDeltaEta = cms.untracked.double(0.08)
tmb.gemMatchDeltaBX = cms.untracked.int32(1)

lct_keep_soft_stubs = False
dphi_lct_pad98 = {
    'pt05' : { 'odd' :   0.0220351 , 'even' :  0.00930056 },
    'pt06' : { 'odd' :   0.0182579 , 'even' :  0.00790009 },
    'pt10' : { 'odd' :     0.01066 , 'even' :  0.00483286 },
    'pt15' : { 'odd' :  0.00722795 , 'even' :   0.0036323 },
    'pt20' : { 'odd' :  0.00562598 , 'even' :  0.00304879 },
    'pt30' : { 'odd' :  0.00416544 , 'even' :  0.00253782 },
    'pt40' : { 'odd' :  0.00342827 , 'even' :  0.00230833 }
}
if ptdphi == 'pt0':
    lct_keep_soft_stubs = True
if lct_keep_soft_stubs:
    tmb.gemClearNomatchLCTs = cms.untracked.bool(False) 
    tmb.gemMatchDeltaPhiOdd = cms.untracked.double(2.)
    tmb.gemMatchDeltaPhiEven = cms.untracked.double(2.)
else:
    tmb.gemMatchDeltaPhiOdd = cms.untracked.double(dphi_lct_pad98[ptdphi]['odd'])
    tmb.gemMatchDeltaPhiEven = cms.untracked.double(dphi_lct_pad98[ptdphi]['even'])

## upgrade CSC TrackFinder
from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_L1TrackFinder
process = customise_csc_L1TrackFinder(process)

## upgrade L1Extra step
from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_L1Extra_allsim
process = customise_csc_L1Extra_allsim(process)
process.l1extraParticles.centralBxOnly = cms.bool(True)
process.l1extraParticles.produceMuonParticles = cms.bool(True)
process.l1extraParticles.produceCaloParticles = cms.bool(False)
process.l1extraParticles.ignoreHtMiss = cms.bool(False)

addPileUp = True
if addPileUp:
    # list of MinBias files for pileup has to be provided
    path = os.getenv( "CMSSW_BASE" ) + "/src/GEMCode/SimMuL1/test/"
    ff = open('%sfilelist_minbias_61M_good.txt'%(path), "r")
    pu_files = ff.read().split('\n')
    ff.close()
    pu_files = filter(lambda x: x.endswith('.root'),  pu_files)

    process.mix.input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
             #### THIS IS AVERAGE PILEUP NUMBER THAT YOU NEED TO CHANGE
            averageNumber = cms.double(pu)
        ),
        type = cms.string('poisson'),
        sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(*pu_files)
    )

## input commands
process.source = cms.Source("PoolSource",
  duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
  inputCommands = cms.untracked.vstring('keep  *_*_*_*'),
  fileNames = cms.untracked.vstring('file:out_digi.root')
)

import os
useInputDir = True
if useInputDir:
    #inputDir = '/pnfs/cms/WAX/11/store/user/lpcgem/dildick/dildick/pT5_1M_v1/DigiL1CSC-MuonGunPt5_1M/82325e40d6202e6fec2dd983c477f3ca/'
    inputDir = '/pnfs/cms/WAX/11/store/user/dildick/dildick/SingleMuPt2-50Fwdv2_1M/SingleMuPt2-50Fwdv2_1M_DIGI_L1CSC_PU140/cadd04af78260520bc45436035aa2787/'
    #inputDir = '/uscms_data/d3/dildick/work/testForInstructions/CMSSW_6_2_0_SLHC1/src/digiFiles/GEM_NeutrinoGun_110K_pu100_DIGI_L1/'
    ls = os.listdir(inputDir)
    process.source.fileNames = cms.untracked.vstring(
        ['file:' + inputDir[:] + x for x in ls if x.endswith('root')]
    )

physics = True
if not physics:
    ## drop all unnecessary collections
    process.source.inputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
        'drop *_simCscTriggerPrimitiveDigis_*_*',
        'drop *_simCsctfTrackDigis_*_*',
        'drop *_simDttfDigis_*_*',
        'drop *_simCsctfDigis_*_*',
        'drop *_simGmtDigis_*_*',
        'drop *_l1extraParticles_*_*'
        )
    
## output commands 
theOutDir = ''
theFileName = 'out_L1.root'
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(theOutDir + theFileName),
    outputCommands = cms.untracked.vstring('keep  *_*_*_*')
)

physics = False
if not physics:
    ## drop all unnecessary collections
    process.output.outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
        # drop all CF stuff
        'drop *_mix_*_*',
        # drop tracker simhits
        'drop PSimHits_*_Tracker*_*',
        # drop calorimetry stuff
        'drop PCaloHits_*_*_*',
        'drop L1Calo*_*_*_*',
        'drop L1Gct*_*_*_*',
        # drop calorimetry l1extra
        'drop l1extraL1Em*_*_*_*',
        'drop l1extraL1Jet*_*_*_*',
        'drop l1extraL1EtMiss*_*_*_*',
        # clean up simhits from other detectors
        'drop PSimHits_*_Totem*_*',
        'drop PSimHits_*_FP420*_*',
        'drop PSimHits_*_BSC*_*',
        # drop some not useful muon digis and links
        'drop *_*_MuonCSCStripDigi_*',
        'drop *_*_MuonCSCStripDigiSimLinks_*',
        'drop *SimLink*_*_*_*',
        'drop *RandomEngineStates_*_*_*',
        'drop *_randomEngineStateProducer_*_*'
        )


## custom sequences
process.mul1 = cms.Sequence(
  process.pdigi *
  process.SimL1MuTriggerPrimitives *
  process.SimL1MuTrackFinders *
  process.simRpcTriggerDigis *
  process.simGmtDigis *
  process.L1Extra
)

process.muL1Short = cms.Sequence(
  process.pdigi *
  process.simCscTriggerPrimitiveDigis *
  process.SimL1MuTrackFinders *
  process.simGmtDigis *
  process.L1Extra
)


## define path-steps
shortRun = False
if shortRun:
    process.l1emu_step      = cms.Path(process.muL1Short)
else: 
    process.l1emu_step      = cms.Path(process.mul1)
process.endjob_step     = cms.Path(process.endOfProcess)
process.out_step        = cms.EndPath(process.output)


## Schedule definition
process.schedule = cms.Schedule(
    process.l1emu_step,
    process.endjob_step,
    process.out_step
)


"""
################################################################################
# Input

import os
ls = os.listdir(inputDir)


process.source = cms.Source("PoolSource",
  duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
  inputCommands = cms.untracked.vstring(
    'keep  *_*_*_*',
    'drop *_simCscTriggerPrimitiveDigis_*_*',
    'drop *_simCsctfTrackDigis_*_*',
    'drop *_simDttfDigis_*_*',
    'drop *_simCsctfDigis_*_*',
    'drop *_simGmtDigis_*_*',
    'drop *_l1extraParticles_*_*'
  ),
  fileNames = cms.untracked.vstring(
    #'file:'+inFile
    #['file:'+inputDir+"/"+x for x in ls if x.endswith('root')]
    #'dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpcgem/khotilov/khotilov/MuomGUN_SIM_Pt2-50_100k/MuonGun_DIGI_L1_Pt2-50_100k/29891ddb18281fff4c42a6e5f5d4bc55/out_22_1_pbQ.root'
    [inputDir[16:] + x for x in ls if x.endswith('root')]
  )
)
"""

"""
################################################################################
# Output

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(theOutDir+theFileName),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
        # drop all CF stuff
        'drop *_mix_*_*',
        # drop tracker simhits
        'drop PSimHits_*_Tracker*_*',
        # drop calorimetry stuff
        'drop PCaloHits_*_*_*',
        'drop L1Calo*_*_*_*',
        'drop L1Gct*_*_*_*',
        # drop calorimetry l1extra
        'drop l1extraL1Em*_*_*_*',
        'drop l1extraL1Jet*_*_*_*',
        'drop l1extraL1EtMiss*_*_*_*',
        # clean up simhits from other detectors
        'drop PSimHits_*_Totem*_*',
        'drop PSimHits_*_FP420*_*',
        'drop PSimHits_*_BSC*_*',
        # drop some not useful muon digis and links
        'drop *_*_MuonCSCStripDigi_*',
        'drop *_*_MuonCSCStripDigiSimLinks_*',
        'drop *SimLink*_*_*_*',
        'drop *RandomEngineStates_*_*_*',
        'drop *_randomEngineStateProducer_*_*'
    )
)
"""


"""
################################################################################
# 

dir_muPt5_pu0 = '/pnfs/cms/WAX/11/store/user/lpcgem/dildick/dildick/pT5_1M_v1/DigiL1CSC-MuonGunPt5_1M/82325e40d6202e6fec2dd983c477f3ca/'
dir_muPt20_pu0 = '/pnfs/cms/WAX/11/store/user/lpcgem/dildick/dildick/pT20_1M_v1/DigiL1CSC-MuonGunPt20_1M/82325e40d6202e6fec2dd983c477f3ca/'
dir_muPt20_pu200 = '/pnfs/cms/WAX/11/store/user/lpcgem/dildick/dildick/muonGun_50k_pT20_lpcgem/DigiL1CSC-MuonGunPt20_50k-PU200/0de68e8f275fa8e4e39f6990099d44a2/'
dir_mb_pu100 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/NeutrinoGunPt5-40_v3/NeutrinoGun_pileup100_Pt5-40_50k_digi/6e54e2af5e7284c73c4665d7969fdf1d/'
dirPt2Pt50 = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/khotilov/MuomGUN_SIM_Pt2-50_100k/MuonGun_DIGI_L1_Pt2-50_100k/29891ddb18281fff4c42a6e5f5d4bc55/'
dirWMu = '/pnfs/cms/WAX/11/store/user/lpcgem/dildick/dildick/WtoMuNu_test/WtoMuNu_test/58b3ea3c731a8657fd489c8437f5f991/'

inputDir = dir_muPt20_pu200
inputDir = dir_muPt20_pu0
inputDir = dir_mb_pu100
inputDir = dirPt2Pt50
#inputDir = dirWMu


theOutDir = ''
theFileName = 'out.root'

min_clct_pattern = 2

gem_match = False
gem_match = True

lct_store_gemdphi = False
#lct_store_gemdphi = True

pt_dphi = 'pt40'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

#process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

from GEMCode.SimMuL1.GEMCSCdPhiLib import *
dphi_lct_pad = dphi_lct_pad98
"""
