import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_43_V4::All'


process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
# process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

# add TrackDetectorAssociator lookup maps to the EventSetup
process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff") 
# from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *  
from TrackingTools.TrackAssociator.default_cfi import * 

process.demo = cms.EDAnalyzer('TestTrackAssociator',
    TrackAssociatorParameterBlock
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    # categories = cms.untracked.vstring('TrackAssociator','TrackAssociatorVerbose'),
    # categories = cms.untracked.vstring('TrackAssociator'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        # threshold = cms.untracked.string('DEBUG'),
	noTimeStamps = cms.untracked.bool(True),
	noLineBreaks = cms.untracked.bool(True),
	DEBUG = cms.untracked.PSet(
           limit = cms.untracked.int32(0)
	),
	# TrackAssociator = cms.untracked.PSet(
	#   limit = cms.untracked.int32(-1)
	#),
	# TrackAssociatorVerbose = cms.untracked.PSet(
	#   limit = cms.untracked.int32(-1)
	#),
    ),
    debugModules = cms.untracked.vstring("demo")
)

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
options = cms.untracked.PSet(
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	'/store/relval/CMSSW_4_3_0_pre6/RelValSingleMuPt10/GEN-SIM-RECO/MC_43_V3-v1/0086/BA16DCB9-2A8C-E011-AD40-0030486791BA.root'
    )
)

process.p = cms.Path(process.demo)


