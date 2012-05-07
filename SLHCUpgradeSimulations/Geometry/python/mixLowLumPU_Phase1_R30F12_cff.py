# Phase 1 R34V25 minbias pileup files
# E34 cm-2s-1
import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup in the design LHC (10**34)
from SimGeneral.MixingModule.mixObjects_cfi import *
mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    bunchspace = cms.int32(25), ## nsec
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
                   
    input = cms.SecSource("PoolSource",
    nbPileupEvents = cms.PSet(
	    averageNumber = cms.double(50.0)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
        # Using one Block from a much larger Dataset.   Block size: 66.6GB, Number of events: 287,400, Number of files: 28
	# /MinBias_TuneZ2star_14TeV-pythia6/Summer12-UpgrdPhase1-v1/GEN-SIM#e4f5c712-88c1-11e1-9dc0-00221959e72f
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/06AB85F0-1189-E111-BD59-001E4F1BC1D4.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/10FE8F2A-DF88-E111-9ED8-001F2965D4EA.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/1ADA1437-0389-E111-A3A6-002590207C28.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/228B1E86-E788-E111-81A2-D48564591BF4.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/02273503-D388-E111-853C-D4856459AC42.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/0A1AF8BA-2A89-E111-A35C-001E0B8DE942.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/465FBACA-3189-E111-B40C-A4BADB3CE8FE.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/6ADDE9C9-2989-E111-96BC-0026B9532A81.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/0E661739-2E89-E111-ACA1-0025902D06D8.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/305CAA84-3C89-E111-AFBF-20CF307C992A.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/3C68B1FC-2289-E111-8A54-001E0B62EBB2.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/7E19C486-3389-E111-B744-0026B9532A81.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/8ACA38BA-3C89-E111-98B1-001E0B62A9E0.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/44A5111B-C488-E111-BF49-001E0B62DBFA.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/709ACC9E-2C89-E111-98B5-001E4F1B8E39.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/78967CD5-2889-E111-B7C6-001E4F1C5820.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/AC6C7FB2-F388-E111-8078-A4BADB3CE8FE.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/7C8F1B4C-FD88-E111-BC1B-001F2965D4EA.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/AEF293BB-3E89-E111-B343-001F29656386.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/B8000132-C188-E111-AB1D-20CF3027A61A.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/D6BBC1E6-1789-E111-9A49-A4BADB3CE43E.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/DE9EAAAA-2F89-E111-BB53-D48564597C70.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/948C96DB-3289-E111-BB1D-20CF3027A5FB.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/9650AC87-3C89-E111-92F1-20CF30725206.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/F20602ED-3C89-E111-9285-20CF30725206.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/A0CD4E59-2D89-E111-A12F-D48564593F96.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/EA29408F-3C89-E111-89ED-20CF307C992A.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdPhase1-v1/0000/ACE74C2B-2889-E111-93D7-00221952AA1F.root'
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
