import FWCore.ParameterSet.Config as cms
process = cms.Process("TripletTest")

import FWCore.ParameterSet.Config as cms

process = cms.Process("TKSEEDING")


#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                   ignoreTotal=cms.untracked.int32(1),
                                   oncePerEventMode=cms.untracked.bool(False)
)


# source
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( ['file:step3.root' ])
secFiles.extend( ['file:step2.root'] )

process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')



#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('*'),
#    destinations = cms.untracked.vstring('cout'),
#    cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
#)

from RecoTracker.PixelSeeding.PixelTripletHLTGenerator_cfi import *
from RecoTracker.PixelSeeding.PixelTripletLargeTipGenerator_cfi import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *

process.triplets = cms.EDAnalyzer("HitTripletProducer",
  OrderedHitsFactoryPSet = cms.PSet(
    ComponentName = cms.string("StandardHitTripletGenerator"),
    SeedingLayers = cms.InputTag("PixelLayerTriplets"),
    GeneratorPSet = cms.PSet( PixelTripletHLTGenerator )
#    GeneratorPSet = cms.PSet( PixelTripletLargeTipGenerator )
  ),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
  )

)

#process.triplets.OrderedHitsFactoryPSet.GeneratorPSet.useFixedPreFiltering = cms.bool(True)
#process.triplets.RegionFactoryPSet.RegionPSet.ptMin = cms.double(1000.00)
#process.triplets.RegionFactoryPSet.RegionPSet.originRadius = cms.double(0.001)
#process.triplets.RegionFactoryPSet.RegionPSet.originHalfLength = cms.double(0.0001)

process.p = cms.Path(process.siPixelRecHits+process.PixelLayerTriplets+process.triplets)
#process.p = cms.Path(process.PixelLayerTriplets+process.triplets)


# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

