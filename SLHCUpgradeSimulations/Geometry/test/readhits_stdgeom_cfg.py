import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessFastsimHitNtuplizer")

# Number of events to be processed
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('file:/uscms_data/d1/cheung/slhc/stdgeom_muon_50GeV.root')
)


# Initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.TrackerDigiGeometryESModule = cms.ESProducer("TrackerDigiGeometryESModule",
    fromDDD = cms.bool(True),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string(''),
    appendToDataLabel = cms.string('')
)

#from Configuration.StandardSequences.Geometry_cff import *
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#TrackerDigiGeometryESModule.applyAlignment = False
#TrackerDigiGeometryESModule.fromDDD = True

process.TrackerGeometricDetESModule = cms.ESProducer("TrackerGeometricDetESModule",
    fromDDD = cms.bool(True)
)
#TrackerGeometricDetESModule.fromDDD = True

process.ReadLocalMeasurement = cms.EDAnalyzer("FastsimHitNtuplizer",
   HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSRecHits"),
   VerbosityLevel = cms.untracked.int32(1),
   OutputFile = cms.string("fsgrechit_ntuple.root")
)

process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")

#process.p = cms.Path(process.contentAna)
process.p = cms.Path(process.ReadLocalMeasurement)
