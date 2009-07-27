# The following comments couldn't be translated into the new config version:

# E33 cm-2s-1
# mb
import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup in the low-luminosity phase
from SimGeneral.MixingModule.mixObjects_cfi import *
mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    bunchspace = cms.int32(25), ## nsec
    checktof = cms.bool(False),
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
                   
    input = cms.SecSource("PoolSource",
	nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(2.8)
        ),
        seed = cms.int32(1234567),
        type = cms.string('poisson'),
	sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre11/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/ECDB1818-A964-DE11-9B4B-001D09F24934.root',
        '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/D245A5BB-4C64-DE11-9F79-001D09F248F8.root',
        '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/C65577F4-EC64-DE11-8D4A-001D09F251CC.root',
        '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/965505C4-9264-DE11-A3BC-001D09F232B9.root',
        '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/5E309A39-7264-DE11-978E-001D09F2A690.root'
	)
    ),
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


