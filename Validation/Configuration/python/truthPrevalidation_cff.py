# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.truthGraphProducer_cfi import truthGraphProducer

truthLogicalGraphProducer = cms.EDProducer(
    "TruthLogicalGraphProducer",

    src = cms.InputTag("truthGraphProducer"),

    simTracks = cms.InputTag("g4SimHits"),
    simVertices = cms.InputTag("g4SimHits"),

    genEventHepMC3 = cms.InputTag("generatorSmeared"),
    genEventHepMC = cms.InputTag("generatorSmeared"),

    mergeGenSimVertices = cms.bool(True),

    postProcessing = cms.PSet(
        collapseIntermediateGenParticles = cms.bool(True),
        seedPdgIds = cms.vint32(),
        seedHadronFlavors = cms.vint32(),
        seedParentDepth = cms.uint32(0),
        keepStableSpectators = cms.bool(True),
        decayPdgIdGroups = cms.VPSet(),
        ignoredPdgIds = cms.vint32(),
        ignoredParticleIds = cms.vuint32(),

        mergeGenSimVerticesByPosition = cms.bool(True),
        genSimVertexPositionTolerance = cms.double(1e-6),
    ),
)

simHitToRecHitMapProducer = cms.EDProducer(
    "SimHitToRecHitMapProducer",

    hgcalRecHits = cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits"),
    ),

    pfRecHits = cms.VInputTag(
        cms.InputTag("particleFlowRecHitECAL", "Cleaned"),
        cms.InputTag("particleFlowRecHitHBHE", "Cleaned"),
        cms.InputTag("particleFlowRecHitHF", "Cleaned"),
        cms.InputTag("particleFlowRecHitHO", "Cleaned"),
    ),
)

truthLogicalGraphHitIndexProducer = cms.EDProducer(
    "TruthLogicalGraphHitIndexProducer",

    src = cms.InputTag("truthLogicalGraphProducer"),
    rawSrc = cms.InputTag("truthGraphProducer"),
    recHitMap = cms.InputTag("simHitToRecHitMapProducer"),

    simHitCollections = cms.VInputTag(
        cms.InputTag("g4SimHits", "HGCHitsEE"),
        cms.InputTag("g4SimHits", "HGCHitsHEfront"),
        cms.InputTag("g4SimHits", "HGCHitsHEback"),
        cms.InputTag("g4SimHits", "EcalHitsEB"),
        cms.InputTag("g4SimHits", "HcalHits"),
    ),

    doHGCalRelabelling = cms.bool(False),
)

truthGraphDumper = cms.EDAnalyzer(
    "TruthGraphDumper",
    src=cms.InputTag("truthGraphProducer"),
    dotFile=cms.string("truthgraph.dot"), # output file
    maxNodes=cms.uint32(20000),
    maxEdgesPerNode=cms.uint32(50),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    genEventHepMC=cms.InputTag("generatorSmeared"),
    genEventHepMC3=cms.InputTag("generatorSmeared"),
)


truthLogicalGraphDumper = cms.EDAnalyzer(
    "TruthLogicalGraphDumper",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),

    hgcalRecHits=cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits", "RECO"),
    ),

    pfRecHits=cms.VInputTag(
        cms.InputTag("particleFlowRecHitECAL", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHBHE", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHF", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHO", "Cleaned", "RECO"),
    ),

    dotFile=cms.string("truthlogicalgraph.dot"), # output file

    maxParticles=cms.uint32(20000),
    maxVertices=cms.uint32(20000),
    maxEdgesPerNode=cms.uint32(300),

    hideLargeSimSourceVertices=cms.bool(True),
    largeSimSourceVertexMinOutgoing=cms.uint32(50),

    hideZeroSimHitSubgraphs=cms.bool(True),
)

truthGraphPrevalidation = cms.Sequence(
    truthGraphProducer
  + truthLogicalGraphProducer
  + simHitToRecHitMapProducer
  + truthLogicalGraphHitIndexProducer
)