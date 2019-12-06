import FWCore.ParameterSet.Config as cms

caloParticles = cms.PSet(
	accumulatorType = cms.string('CaloTruthAccumulator'),
	#createUnmergedCollection = cms.bool(True),
	#createMergedBremsstrahlung = cms.bool(True),
	#createInitialVertexCollection = cms.bool(False),
	#alwaysAddAncestors = cms.bool(True),
        MinEnergy = cms.double(0.5),
        MaxPseudoRapidity = cms.double(5.0),
        premixStage1 = cms.bool(False),
	maximumPreviousBunchCrossing = cms.uint32(0),
	maximumSubsequentBunchCrossing = cms.uint32(0),
	simHitCollections = cms.PSet(
#            hgc = cms.VInputTag(
#                cms.InputTag('g4SimHits','HGCHitsEE'),
#                cms.InputTag('g4SimHits','HGCHitsHEfront'),
#                cms.InputTag('g4SimHits','HcalHits')
#            ),
#            hcal = cms.VInputTag(cms.InputTag('g4SimHits','HcalHits')),
            ecal = cms.VInputTag(
                cms.InputTag('g4SimHits','EcalHitsEE'),
                cms.InputTag('g4SimHits','EcalHitsEB'),
                cms.InputTag('g4SimHits','EcalHitsES')
            )
	),
	simTrackCollection = cms.InputTag('g4SimHits'),
	simVertexCollection = cms.InputTag('g4SimHits'),
	genParticleCollection = cms.InputTag('genParticles'),
	allowDifferentSimHitProcesses = cms.bool(False), # should be True for FastSim, False for FullSim
	HepMCProductLabel = cms.InputTag('generatorSmeared'),
)

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(caloParticles, premixStage1 = True)

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
phase2_hgcalV9.toModify(
    caloParticles,
    simHitCollections = dict(hgc = {2 : cms.InputTag('g4SimHits','HGCHitsHEback')} ),
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(caloParticles, cms.PSet()) # don't allow this to run in fastsim

