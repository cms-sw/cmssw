import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects
from SimGeneral.MixingModule.mixPoolSource_cfi import *
from SimGeneral.MixingModule.digitizers_cfi import *

FileNames = cms.untracked.vstring(['/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/16547ECB-9C4B-E411-A815-0025905964BC.root', '/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/86C3C326-9F4B-E411-903D-0025905A48EC.root', '/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/C48D8223-9F4B-E411-BC37-0026189438DC.root', '/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/D070AB62-9D4B-E411-9766-002618FDA207.root'])

mixGen = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(0),
    minBunch = cms.int32(0), ## in terms of 25 nsec
    bunchspace = cms.int32(1), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.0)
        ),
        type = cms.string('fixed'),
                          sequential = cms.untracked.bool(False),
        fileNames = FileNames
    ),

    mixObjects = cms.PSet(
        mixHepMC = cms.PSet(
            input = cms.VInputTag(cms.InputTag("generator","unsmeared"),cms.InputTag("generator")),
            makeCrossingFrame = cms.untracked.bool(True),
            type = cms.string('HepMCProduct')
            )
        ),
)

# How to??
#for a in self.aliases: delattr(self, a)
# here??

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(theDigitizers),
    LabelPlayback = cms.string('mixGen'),
    maxBunch = cms.int32(0),
    minBunch = cms.int32(0), ## in terms of 25 nsec

    bunchspace = cms.int32(1), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(True),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.0)
        ),
        type = cms.string('fixed'),
                          sequential = cms.untracked.bool(False),
        fileNames = FileNames
    ),
    mixObjects = cms.PSet(theMixObjects)
)

mixVal = mixGen.clone(playback = cms.untracked.bool(True),LabelPlayback = cms.string('mixGen'))

