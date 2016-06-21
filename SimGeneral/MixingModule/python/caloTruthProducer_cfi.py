import FWCore.ParameterSet.Config as cms

caloParticles = cms.PSet(
	accumulatorType = cms.string('CaloTruthAccumulator'),
#	createUnmergedCollection = cms.bool(True),
#	createMergedBremsstrahlung = cms.bool(True),
#	createInitialVertexCollection = cms.bool(False),
#	alwaysAddAncestors = cms.bool(True),
	maximumPreviousBunchCrossing = cms.uint32(0),
	maximumSubsequentBunchCrossing = cms.uint32(0),
	simHitCollections = cms.PSet(
            hgc = cms.VInputTag(
                cms.InputTag('g4SimHits','HGCHitsEE'),
                cms.InputTag('g4SimHits','HGCHitsHEfront'),
#                cms.InputTag('g4SimHits','HGCHitsHEback')
            ),
#            hcal = cms.VInputTag(cms.InputTag('g4SimHits','HcalHits')),
#            ecal = cms.VInputTag(
#                cms.InputTag('g4SimHits','EcalHitsEE'),
#                cms.InputTag('g4SimHits','EcalHitsEB'),
#                cms.InputTag('g4SimHits','EcalHitsES')
#            )
	),
	simTrackCollection = cms.InputTag('g4SimHits'),
	simVertexCollection = cms.InputTag('g4SimHits'),
	genParticleCollection = cms.InputTag('genParticles'),
	allowDifferentSimHitProcesses = cms.bool(False), # should be True for FastSim, False for FullSim
	HepMCProductLabel = cms.InputTag('generatorSmeared')
)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    raise Exception,"Not translated for fast sim"
    # for unknown reasons, fastsim needs this flag on
    caloParticles.allowDifferentSimHitProcesses = True
    # fastsim labels for simhits, simtracks, simvertices
    caloParticles.simHitCollections = cms.PSet(
        # I dunno what would go here
        )
    caloParticles.simTrackCollection = cms.InputTag('famosSimHits')
    caloParticles.simVertexCollection = cms.InputTag('famosSimHits')
