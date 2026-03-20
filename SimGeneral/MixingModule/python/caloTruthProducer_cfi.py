import FWCore.ParameterSet.Config as cms

run3_simHits = {
    "ECAL": [
        cms.InputTag('g4SimHits', 'EcalHitsEB'),
        cms.InputTag('g4SimHits', 'EcalHitsEE'),
    ],
    "HCAL": [
        cms.InputTag('g4SimHits', 'HcalHits'),
    ],
}

ph2_simHits = {
    "ECAL": [
        cms.InputTag('g4SimHits', 'EcalHitsEB'),
    ],
    "HCAL": [
        cms.InputTag('g4SimHits', 'HcalHits'),
    ],
    "HGCAL": [
        cms.InputTag('g4SimHits', 'HGCHitsEE'),
        cms.InputTag('g4SimHits', 'HGCHitsHEfront'),
        cms.InputTag('g4SimHits', 'HGCHitsHEback'),
    ]
}

# Default run-3 configuration
caloParticles = cms.PSet(
	accumulatorType = cms.string('CaloTruthAccumulator'),
#	createUnmergedCollection = cms.bool(True),
#	createMergedBremsstrahlung = cms.bool(True),
#	createInitialVertexCollection = cms.bool(False),
#	alwaysAddAncestors = cms.bool(True),
    MinEnergy = cms.double(0.5),
    MaxPseudoRapidity = cms.double(5.0),
    premixStage1 = cms.bool(False),
    doHGCAL = cms.bool(False),
	maximumPreviousBunchCrossing = cms.uint32(0),
	maximumSubsequentBunchCrossing = cms.uint32(0),
	simHitCollections = cms.PSet(
        hgc = cms.VInputTag(),
        hcal = cms.VInputTag(*run3_simHits["HCAL"]),
        ecal = cms.VInputTag(*run3_simHits["ECAL"]),
	),
	simTrackCollection = cms.InputTag('g4SimHits'),
	simVertexCollection = cms.InputTag('g4SimHits'),
	genParticleCollection = cms.InputTag('genParticles'),
	allowDifferentSimHitProcesses = cms.bool(False), # should be True for FastSim, False for FullSim
	HepMCProductLabel = cms.InputTag('generatorSmeared'),
)


# Phase-2 configuration (HGCAL only) # [FIXME: with the isBarrel check in associators it could contain all hits]
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(caloParticles, 
    doHGCAL=True,
    simHitCollections = cms.PSet(
        hgc = cms.VInputTag(*ph2_simHits["HGCAL"]),
        hcal = cms.VInputTag(*ph2_simHits["HCAL"]),
        ecal = cms.VInputTag(*ph2_simHits["ECAL"]),
    ),
)

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
phase2_hfnose.toModify(
    caloParticles,
    simHitCollections = dict(
        hgc = caloParticles.simHitCollections.hgc + [cms.InputTag('g4SimHits','HFNoseHits')],
#        hcal = cms.VInputTag(cms.InputTag('g4SimHits','HcalHits'))
    )
)

from Configuration.ProcessModifiers.run3_ecalclustering_cff import run3_ecalclustering
run3_ecalclustering.toModify(
        caloParticles,
	simHitCollections = cms.PSet(
            ecal = cms.VInputTag(
                cms.InputTag('g4SimHits','EcalHitsEE'),
                cms.InputTag('g4SimHits','EcalHitsEB'),
                cms.InputTag('g4SimHits','EcalHitsES')
            )
	)
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toModify(
    caloParticles, 
    simHitCollections = cms.PSet(
        hgc = cms.VInputTag(*ph2_simHits["HGCAL"]),
        hcal = cms.VInputTag(*ph2_simHits["HCAL"]),
        ecal = cms.VInputTag(*ph2_simHits["ECAL"]),
    )
)

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(caloParticles, premixStage1 = True)
