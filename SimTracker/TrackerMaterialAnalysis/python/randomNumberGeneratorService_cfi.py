# The following comments couldn't be translated into the new config version:

# Random number generators for (optional) modules up to trackingMaterialProducer

import FWCore.ParameterSet.Config as cms

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        trackingMaterialProducer = cms.untracked.uint32(288269),
        VtxSmeared = cms.untracked.uint32(863771)
    ),
    sourceSeed = cms.untracked.uint32(442302)
)
