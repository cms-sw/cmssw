# The following comments couldn't be translated into the new config version:

# Random number generators for (optional) modules up to trackingMaterialProducer

import FWCore.ParameterSet.Config as cms

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                           VtxSmeared = cms.PSet(
                                             initialSeed = cms.untracked.uint32(863771),
                                             engineName = cms.untracked.string('HepJamesRandom')
                                             ),
                                           trackingMaterialProducer = cms.PSet(
                                             initialSeed = cms.untracked.uint32(288269),
                                             engineName = cms.untracked.string('HepJamesRandom')
                                             ),
                                           generator = cms.PSet(
                                             initialSeed = cms.untracked.uint32(220675),
                                             engineName = cms.untracked.string('HepJamesRandom')
                                             )
                                           )
