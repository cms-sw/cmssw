############################################################
# define basic process
############################################################

import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TrackNtuple")
 

############################################################
# load global tag
############################################################

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')


############################################################
# logger, input & output files
############################################################

### LongBarrel geometry
process.load('Configuration.Geometry.GeometryExtendedPhase2TkLB6PS_cff')

process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.load("FWCore.MessageService.MessageLogger_cfi")

#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
 
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(
    "/store/group/comm_trigger/L1TrackTrigger/6_1_2_SLHC2/muDST/Muon/LB6PS/Muon_LB6PS_v3.root")
    )

process.TFileService = cms.Service("TFileService", fileName = cms.string('SingleMuon_LB_TrkPerf.root'))


############################################################
# my analyzer
############################################################

process.L1TrackNtuple = cms.EDAnalyzer('L1TrackNtupleMaker')
process.ana = cms.Path(process.L1TrackNtuple)

############################################################
# schedule
############################################################

process.schedule = cms.Schedule(process.ana)

