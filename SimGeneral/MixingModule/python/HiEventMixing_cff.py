import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixObjects_cfi import *
eventEmbeddingSourceParameters = cms.PSet(
    nbPileupEvents = cms.PSet(
        averageNumber = cms.double(1.0)
    ),
    seed = cms.int32(325),
    type = cms.string('fixed'),
    sequential = cms.untracked.bool(True)
)
eventEmbeddingMixParameters = cms.PSet(
    LabelPlayback = cms.string(''),
    playback = cms.untracked.bool(False),
    maxBunch = cms.int32(0),
    minBunch = cms.int32(0),
    Label = cms.string(''),
    bunchspace = cms.int32(125),
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),
    useCurrentProcessOnly = cms.bool(False)
    )
simEventEmbeddingMixParameters = cms.PSet(
    eventEmbeddingMixParameters,
    mixObjects = cms.PSet(
        mixCH = cms.PSet(
            mixCaloHits
        ),
        mixTracks = cms.PSet(
            mixSimTracks
        ),
        mixVertices = cms.PSet(
            mixSimVertices
        ),
        mixSH = cms.PSet(
            mixSimHits
        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)
genEventEmbeddingMixParameters = cms.PSet(
    eventEmbeddingMixParameters,
    mixObjects = cms.PSet(
        mySet = cms.PSet(
            input = cms.VInputTag(cms.InputTag("generator"), cms.InputTag("secsource")),
            type = cms.string('HepMCProduct')
        )
    )
)

mixSim = cms.EDProducer("MixingModule",
                        simEventEmbeddingMixParameters,
                        input = cms.SecSource("PoolRASource",
                                              eventEmbeddingSourceParameters,
                                              fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/cmshi/mc/sim/hydjet_sim_x2_c10_d20080425/hydjet_sim_x2_c10_d20080425_r000002.root')
                                              )
                             )


mixGen = cms.EDProducer("MixingModule",
                        genEventEmbeddingMixParameters,
                        input = cms.SecSource("PoolRASource",
                                              eventEmbeddingSourceParameters,
                                              fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/cmshi/mc/sim/hydjet_sim_x2_c10_d20080425/hydjet_sim_x2_c10_d20080425_r000002.root')
                                              )
                        )



mixGenNoPU = cms.EDProducer("MixingModule",
                            genEventEmbeddingMixParameters)

mixSimNoPU = cms.EDProducer("MixingModule",
                            simEventEmbeddingMixParameters)

#Parameters for Signal-Only digitization in Heavy Ion Mixing

noMix = mixSimNoPU.clone()
noMix.mixObjects.mixHepMC.input = cms.VInputTag(cms.InputTag("hiSignal"))

noMix.mixObjects.mixCH.input = cms.VInputTag(cms.InputTag("hiSignalG4SimHits","CaloHitsTk"), cms.InputTag("hiSignalG4SimHits","CastorBU"),
                                             cms.InputTag("hiSignalG4SimHits","CastorFI"), cms.InputTag("hiSignalG4SimHits","CastorPL"), cms.InputTag("hiSignalG4SimHits","CastorTU"),
                                             cms.InputTag("hiSignalG4SimHits","EcalHitsEB"), cms.InputTag("hiSignalG4SimHits","EcalHitsEE"), cms.InputTag("hiSignalG4SimHits","EcalHitsES"),
                                             cms.InputTag("hiSignalG4SimHits","EcalTBH4BeamHits"), cms.InputTag("hiSignalG4SimHits","HcalHits"),
                                             cms.InputTag("hiSignalG4SimHits","HcalTB06BeamHits"), cms.InputTag("hiSignalG4SimHits","ZDCHITS"))

noMix.mixObjects.mixSH.input = cms.VInputTag(cms.InputTag("hiSignalG4SimHits","BSCHits"), cms.InputTag("hiSignalG4SimHits","FP420SI"), cms.InputTag("hiSignalG4SimHits","MuonCSCHits"),
                                             cms.InputTag("hiSignalG4SimHits","MuonDTHits"), cms.InputTag("hiSignalG4SimHits","MuonRPCHits"),
                                             cms.InputTag("hiSignalG4SimHits","TotemHitsRP"), cms.InputTag("hiSignalG4SimHits","TotemHitsT1"),
                                             cms.InputTag("hiSignalG4SimHits","TotemHitsT2Gem"), cms.InputTag("hiSignalG4SimHits","TrackerHitsPixelBarrelHighTof"),
                                             cms.InputTag("hiSignalG4SimHits","TrackerHitsPixelBarrelLowTof"), cms.InputTag("hiSignalG4SimHits","TrackerHitsPixelEndcapHighTof"),
                                             cms.InputTag("hiSignalG4SimHits","TrackerHitsPixelEndcapLowTof"), cms.InputTag("hiSignalG4SimHits","TrackerHitsTECHighTof"),
                                             cms.InputTag("hiSignalG4SimHits","TrackerHitsTECLowTof"), cms.InputTag("hiSignalG4SimHits","TrackerHitsTIBHighTof"),
                                             cms.InputTag("hiSignalG4SimHits","TrackerHitsTIBLowTof"), cms.InputTag("hiSignalG4SimHits","TrackerHitsTIDHighTof"),
                                             cms.InputTag("hiSignalG4SimHits","TrackerHitsTIDLowTof"), cms.InputTag("hiSignalG4SimHits","TrackerHitsTOBHighTof"),
                                             cms.InputTag("hiSignalG4SimHits","TrackerHitsTOBLowTof"))

noMix.mixObjects.mixTracks.input = cms.VInputTag(cms.InputTag("hiSignalG4SimHits"))
noMix.mixObjects.mixVertices.input = cms.VInputTag(cms.InputTag("hiSignalG4SimHits"))


mixGenHI = cms.EDProducer("HiMixingModule",
                          genEventEmbeddingMixParameters,
                          signalTag = cms.vstring("hiSignal","hiSignalG4SimHits"),
                          srcGEN = cms.vstring("hiSignal","generator")
                          )

mix = cms.EDProducer("HiMixingModule",
                     simEventEmbeddingMixParameters,
                     signalTag = cms.vstring("hiSignal","hiSignalG4SimHits"),
                     srcGEN = cms.vstring("hiSignal","generator"),
                     srcSIM = cms.vstring("hiSignalG4SimHits","g4SimHits")
                     )
