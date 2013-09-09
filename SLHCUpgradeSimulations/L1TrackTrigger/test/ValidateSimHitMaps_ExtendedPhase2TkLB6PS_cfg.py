####################################################
# SLHCUpgradeSimulations                           #
# Configuration file for Full Workflow             #
# Step 2 (again)                                   #
# Understand if everything is fine with            #
# L1TkCluster e L1TkStub                           #
####################################################
# Nicola Pozzobon                                  #
# CERN, August 2012                                #
####################################################

#################################################################################################
# import of general framework
#################################################################################################
import FWCore.ParameterSet.Config as cms
import os
process = cms.Process('ValidateSimHitMaps')

#################################################################################################
# global tag
#################################################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

#################################################################################################
# load the specific tracker geometry
#################################################################################################
process.load('Configuration.Geometry.GeometryExtendedPhase2TkLB6PSReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkLB6PS_cff')
#process.load("SLHCUpgradeSimulations.Utilities.StackedTrackerGeometry_cfi")

#process.TrackerDigiGeometryESModule.applyAlignment = False
#process.TrackerDigiGeometryESModule.fromDDD = cms.bool(True)
#process.TrackerGeometricDetESModule.fromDDD = cms.bool(True)

#################################################################################################
# load the magnetic field
#################################################################################################
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
#process.load('Configuration.StandardSequences.MagneticField_40T_cff')
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

#################################################################################################
# define the source and maximum number of events to generate and simulate
#################################################################################################
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
     fileNames = cms.untracked.vstring('file:TenMuPt_0_50_ExtendedPhase2TkLB6PS_5000_GEN_SIM.root')
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(3000)
)

#################################################################################################
# load the analyzer
#################################################################################################
process.ValidateSimHitMaps = cms.EDAnalyzer("ValidateSimHitMaps",
#    DebugMode = cms.bool(True)
)

#################################################################################################
# define output file and message logger
#################################################################################################
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('file:ValidateSimHitMaps_ExtendedPhase2TkLB6PS.root')
)

#################################################################################################
# define the final path to be fed to cmsRun
#################################################################################################
process.p = cms.Path( process.ValidateSimHitMaps )

