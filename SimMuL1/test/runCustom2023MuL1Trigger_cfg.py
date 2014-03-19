## This configuration runs the DIGI+L1Emulator step
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("MUTRG")

## Standard sequence
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load("Configuration.StandardSequences.L1Emulator_cff")
process.load("Configuration.StandardSequences.L1Extra_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

################### Take inputs from crab.cfg file ##############
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.register ('pu',
                  140,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.float,
                  "PU: 100  default")

options.register ('ptdphi',
                  'pt0',
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.string,
                  "ptdphi: 0 GeV/c default")

import sys

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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## GEM geometry customization
from Geometry.GEMGeometry.gemGeometryCustoms import custom_GE11_6partitions_v1
process = custom_GE11_6partitions_v1(process)

## GEM digitizer
from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_muon_only
process = customize_digi_addGEM_muon_only(process)

## upgrade CSC geometry 
from SLHCUpgradeSimulations.Configuration.muonCustoms import unganged_me1a_geometry
process = unganged_me1a_geometry(process)

## upgrade CSC digitizer
from SLHCUpgradeSimulations.Configuration.muonCustoms import digitizer_timing_pre3_median
process = digitizer_timing_pre3_median(process)

## upgrade CSC L1 customizations
from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_L1Stubs 
process = customise_csc_L1Stubs(process)

## GEM-CSC emulator
from SLHCUpgradeSimulations.Configuration.gemCustoms import customise_L1Emulator as customise_L1EmulatorGEM
process = customise_L1EmulatorGEM(process, ptdphi)
process.simCscTriggerPrimitiveDigis.clctSLHC.clctNplanesHitPattern = 3
tmb = process.simCscTriggerPrimitiveDigis.tmbSLHC

"""
tmb.clctToAlct = cms.untracked.bool(False)
tmb.printAvailablePads = cms.untracked.bool(False)
tmb.dropLowQualityCLCTsNoGEMs_ME1a = cms.untracked.bool(True)
tmb.dropLowQualityCLCTsNoGEMs_ME1b = cms.untracked.bool(True)
tmb.buildLCTfromALCTandGEM_ME1a = cms.untracked.bool(False)
tmb.buildLCTfromALCTandGEM_ME1b = cms.untracked.bool(True)
tmb.doLCTGhostBustingWithGEMs = cms.untracked.bool(False)
"""

## RPC-CSC emulator
from SLHCUpgradeSimulations.Configuration.rpcCustoms import customise_L1Emulator as customise_L1EmulatorRPC
process = customise_L1EmulatorRPC(process)

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

## add pile-up to the digi step
from GEMCode.GEMValidation.InputFileHelpers import addPileUp
process = addPileUp(process, pu)

## input commands
process.source = cms.Source("PoolSource",
  duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
  inputCommands = cms.untracked.vstring('keep  *_*_*_*'),
  fileNames = cms.untracked.vstring('file:out_digi.root')
)

## input
from GEMCode.SimMuL1.GEMCSCTriggerSamplesLib import *
from GEMCode.GEMValidation.InputFileHelpers import *
#process = useInputDir(process, eosfiles['_pt2-50_PU140_6part2019'], True)


physics = False
if not physics:
    ## drop all unnecessary collections
    process.source.inputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
        'drop *_simCscTriggerPrimitiveDigis_*_*',
        'drop *_simDtTriggerPrimitiveDigis_*_*',
        'drop *_simRpcTriggerDigis_*_*',
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
#  process.pdigi *
  process.SimL1MuTriggerPrimitives *
  process.SimL1MuTrackFinders *
  process.simRpcTriggerDigis *
  process.simGmtDigis *
  process.L1Extra
)

process.muL1Short = cms.Sequence(
#  process.pdigi *
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

## messages
print
print 'Input files:'
print '----------------------------------------'
print process.source.fileNames
print
print 'Output file:'
print '----------------------------------------'
print process.output.fileName
print 
