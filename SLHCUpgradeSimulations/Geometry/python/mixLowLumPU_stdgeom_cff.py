import FWCore.ParameterSet.Config as cms

mixingFiles = cms.untracked.vstring() 
mixingFiles.extend( [
	# Using one Block from a much larger Dataset.   Block size: 65.6GB, Number of events: 274,800, Number of files: 23
	# /MinBias_TuneZ2star_14TeV-pythia6/Summer12-UpgrdStdGeom-v1/GEN-SIM#3aadca6c-8947-11e1-9dc0-00221959e72f
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/FA806C6D-4889-E111-A202-00259029E670.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/F00999DC-6489-E111-AEDB-00259048A8F4.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/EED1EB46-5F89-E111-838D-00259048AE00.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/EC97B0BE-5E89-E111-916A-00259029ED64.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/EAAD21EE-4589-E111-B143-00259048A8F4.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/E67A1BD8-6689-E111-8832-0025901ABD2E.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/CE51A078-4689-E111-88F3-00304867FD67.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/CC8DABC8-5689-E111-83C6-00259029ED0E.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/C8262F49-4589-E111-96B5-00304867FE1F.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/B487E3FD-4889-E111-9349-00304867FD83.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/9899572A-5D89-E111-ADD2-00259048A87C.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/90D2EF67-6089-E111-9B88-00259048AE50.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/8EE02DB4-4789-E111-BAAC-00259048AC10.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/8E77FAAC-6389-E111-9C7C-00259048AE52.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/847F9024-9E89-E111-A81C-00259019A41A.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/8237A98E-5B89-E111-B01C-0025901AC0FA.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/7C6CB101-6689-E111-AC98-00259048AC9A.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/7830DD50-6289-E111-BAFD-00259048AE52.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/70998A41-4A89-E111-BBA7-00304867FE3C.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/706D0C05-5E89-E111-8687-00259029E66E.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/6CBFF5D0-5F89-E111-A1CE-00259019A41C.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/58BDEAAF-6089-E111-A59F-0025901AC0FC.root',
	'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgrdStdGeom-v1/0000/1E3B82B0-6189-E111-AFB4-00259029ED64.root'
	] );

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

    input = cms.SecSource("EmbeddedRootSource",
    nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(19.7)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = mixingFiles
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
