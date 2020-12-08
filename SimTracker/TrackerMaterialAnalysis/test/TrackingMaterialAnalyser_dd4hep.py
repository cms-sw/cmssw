#! /usr/bin/env cmsRun
"""
   $cmsRun TrackerMaterialAnalyser.py PhaseII
   
   Input: "material.root" produced by:

   $cmsRun MaterialProducer10GeVNeutrino.py
"""
import sys
import FWCore.ParameterSet.Config as cms

process = cms.Process("MaterialAnalyser")

ph1Xml = "dd4hep_trackingMaterialGroups_ForPhaseI.xml"
ph2Xml = "dd4hep_trackingMaterialGroups_ForPhaseII.xml"

groupsPh1 = cms.vstring(
    "TrackerRecMaterialPixelBarrelLayer0_External",
    "TrackerRecMaterialPixelBarrelLayer1_External",
    "TrackerRecMaterialPixelBarrelLayer2_External",
    "TrackerRecMaterialPixelBarrelLayer3_External",
    "TrackerRecMaterialPixelBarrelLayer0",
    "TrackerRecMaterialPixelBarrelLayer1",
    "TrackerRecMaterialPixelBarrelLayer2",
    "TrackerRecMaterialPixelBarrelLayer3",
    "TrackerRecMaterialTIBLayer0_Z0",
    "TrackerRecMaterialTIBLayer0_Z20",
    "TrackerRecMaterialTIBLayer1_Z0",
    "TrackerRecMaterialTIBLayer1_Z30",
    "TrackerRecMaterialTIBLayer2_Z0",
    "TrackerRecMaterialTIBLayer2_Z40",
    "TrackerRecMaterialTIBLayer3_Z0",
    "TrackerRecMaterialTIBLayer3_Z50",
    "TrackerRecMaterialTOBLayer0_Z0",
    "TrackerRecMaterialTOBLayer0_Z20",
    "TrackerRecMaterialTOBLayer0_Z70",
    "TrackerRecMaterialTOBLayer1_Z0",
    "TrackerRecMaterialTOBLayer1_Z20",
    "TrackerRecMaterialTOBLayer1_Z80",
    "TrackerRecMaterialTOBLayer2_Z0",
    "TrackerRecMaterialTOBLayer2_Z25",
    "TrackerRecMaterialTOBLayer2_Z80",
    "TrackerRecMaterialTOBLayer3_Z0",
    "TrackerRecMaterialTOBLayer3_Z25",
    "TrackerRecMaterialTOBLayer3_Z80",
    "TrackerRecMaterialTOBLayer4_Z0",
    "TrackerRecMaterialTOBLayer4_Z25",
    "TrackerRecMaterialTOBLayer4_Z80",
    "TrackerRecMaterialTOBLayer5_Z0",
    "TrackerRecMaterialTOBLayer5_Z25",
    "TrackerRecMaterialTOBLayer5_Z80",
    "TrackerRecMaterialPixelEndcapDisk1Fw_Inner",
    "TrackerRecMaterialPixelEndcapDisk1Fw_Outer",
    "TrackerRecMaterialPixelEndcapDisk2Fw_Inner",
    "TrackerRecMaterialPixelEndcapDisk2Fw_Outer",
    "TrackerRecMaterialPixelEndcapDisk3Fw_Inner",
    "TrackerRecMaterialPixelEndcapDisk3Fw_Outer",
    "TrackerRecMaterialPixelEndcapDisk1Bw_Inner",
    "TrackerRecMaterialPixelEndcapDisk1Bw_Outer",
    "TrackerRecMaterialPixelEndcapDisk2Bw_Inner",
    "TrackerRecMaterialPixelEndcapDisk2Bw_Outer",
    "TrackerRecMaterialPixelEndcapDisk3Bw_Inner",
    "TrackerRecMaterialPixelEndcapDisk3Bw_Outer",
    "TrackerRecMaterialTIDDisk1_R0",
    "TrackerRecMaterialTIDDisk1_R30",
    "TrackerRecMaterialTIDDisk2_R25",
    "TrackerRecMaterialTIDDisk2_R30",
    "TrackerRecMaterialTIDDisk2_R40",
    "TrackerRecMaterialTIDDisk3_R24",
    "TrackerRecMaterialTECDisk0_R20",
    "TrackerRecMaterialTECDisk0_R40",
    "TrackerRecMaterialTECDisk0_R50",
    "TrackerRecMaterialTECDisk0_R60",
    "TrackerRecMaterialTECDisk0_R90",
    "TrackerRecMaterialTECDisk1_R20",
    "TrackerRecMaterialTECDisk2_R20",
    "TrackerRecMaterialTECDisk3",
    "TrackerRecMaterialTECDisk4_R33",
    "TrackerRecMaterialTECDisk5_R33",
    "TrackerRecMaterialTECDisk6",
    "TrackerRecMaterialTECDisk7_R40",
    "TrackerRecMaterialTECDisk8",
)

groupsPh2 = cms.vstring(
    "TrackerRecMaterialPhase2PixelBarrelLayer1",
    "TrackerRecMaterialPhase2PixelBarrelLayer2",
    "TrackerRecMaterialPhase2PixelBarrelLayer3",
    "TrackerRecMaterialPhase2PixelBarrelLayer4",
    "TrackerRecMaterialPhase2PixelForwardDisk1",
    "TrackerRecMaterialPhase2PixelForwardDisk2",
    "TrackerRecMaterialPhase2PixelForwardDisk3",
    "TrackerRecMaterialPhase2PixelForwardDisk4",
    "TrackerRecMaterialPhase2PixelForwardDisk5",
    "TrackerRecMaterialPhase2PixelForwardDisk6",
    "TrackerRecMaterialPhase2PixelForwardDisk7",
    "TrackerRecMaterialPhase2PixelForwardDisk8",
    "TrackerRecMaterialPhase2PixelForwardDisk9",
    "TrackerRecMaterialPhase2PixelForwardDisk10",
    "TrackerRecMaterialPhase2PixelForwardDisk11",
    "TrackerRecMaterialPhase2PixelForwardDisk12",
    "TrackerRecMaterialPhase2OTBarrelLayer1",
    "TrackerRecMaterialPhase2OTBarrelLayer2",
    "TrackerRecMaterialPhase2OTBarrelLayer3",
    "TrackerRecMaterialPhase2OTBarrelLayer4",
    "TrackerRecMaterialPhase2OTBarrelLayer5",
    "TrackerRecMaterialPhase2OTBarrelLayer6",
    "TrackerRecMaterialPhase2OTForwardDisk1",
    "TrackerRecMaterialPhase2OTForwardDisk2",
    "TrackerRecMaterialPhase2OTForwardDisk3",
    "TrackerRecMaterialPhase2OTForwardDisk4",
    "TrackerRecMaterialPhase2OTForwardDisk5"
)

optPh = str(sys.argv[2])

groups = None
if( optPh.lower() == "phasei"):
    groups = groupsPh1
    process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')
elif( optPh.lower() == "phaseii"):
    groups = groupsPh2
else:
    print("Valid options: PhaseI, PhaseII")

from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff import *
from Geometry.EcalCommonData.ecalSimulationParameters_cff import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cff import *
from Geometry.HGCalCommonData.hgcalParametersInitialization_cfi import *
from Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi import *
from Geometry.MuonNumbering.muonGeometryConstants_cff import *
from Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff import *

process.load('FWCore.MessageService.MessageLogger_cfi')

process.DDCompactViewESProducer = cms.ESProducer(
  "DDCompactViewESProducer",
  appendToDataLabel = cms.string('')
)

process.trackingMaterialAnalyser = cms.EDAnalyzer(
    "DD4hep_TrackingMaterialAnalyser",
    MaterialAccounting = cms.InputTag("trackingMaterialProducer"),
    SplitMode = cms.string("NearestLayer"),
    DDDetector = cms.ESInputTag("",""),
    SkipBeforeFirstDetector = cms.bool(False),
    SkipAfterLastDetector = cms.bool(True),
    SaveSummaryPlot = cms.bool(True),
    SaveDetailedPlots = cms.bool(True),
    SaveParameters = cms.bool(True),
    SaveXML = cms.bool(True),
    isHGCal = cms.bool(False),
    isHFNose = cms.bool(False),
    Groups = groups
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:material.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger.files.LogTrackingMaterialAnalysis = dict()
process.MessageLogger.TrackingMaterialAnalysis=dict()
process.path = cms.Path(process.trackingMaterialAnalyser)

