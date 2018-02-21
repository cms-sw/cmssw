import FWCore.ParameterSet.Config as cms

simHitTPAssocProducer = cms.EDProducer("SimHitTPAssociationProducer",
  simHitSrc = cms.VInputTag(cms.InputTag('g4SimHits','MuonDTHits'),
                            cms.InputTag('g4SimHits','MuonCSCHits'),
                            cms.InputTag('g4SimHits','MuonRPCHits'),
                            cms.InputTag('g4SimHits','TrackerHitsTIBLowTof'),
                            cms.InputTag('g4SimHits','TrackerHitsTIBHighTof'),
                            cms.InputTag('g4SimHits','TrackerHitsTIDLowTof'),
                            cms.InputTag('g4SimHits','TrackerHitsTIDHighTof'),
                            cms.InputTag('g4SimHits','TrackerHitsTOBLowTof'),
                            cms.InputTag('g4SimHits','TrackerHitsTOBHighTof'),
                            cms.InputTag('g4SimHits','TrackerHitsTECLowTof'),
                            cms.InputTag('g4SimHits','TrackerHitsTECHighTof'),
                            cms.InputTag( 'g4SimHits','TrackerHitsPixelBarrelLowTof'),
                            cms.InputTag('g4SimHits','TrackerHitsPixelBarrelHighTof'),
                            cms.InputTag('g4SimHits','TrackerHitsPixelEndcapLowTof'),
                            cms.InputTag('g4SimHits','TrackerHitsPixelEndcapHighTof') ),
  trackingParticleSrc = cms.InputTag('mix', 'MergedTrackTruth')
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(simHitTPAssocProducer, simHitSrc = ["g4SimHits:TrackerHitsPixelBarrelLowTof", "g4SimHits:TrackerHitsPixelEndcapLowTof"])

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(simHitTPAssocProducer,
    simHitSrc = ["fastSimProducer:TrackerHits",
                 "MuonSimHits:MuonCSCHits",
                 "MuonSimHits:MuonDTHits",
                 "MuonSimHits:MuonRPCHits"]
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(simHitTPAssocProducer, trackingParticleSrc = "mixData:MergedTrackTruth")
