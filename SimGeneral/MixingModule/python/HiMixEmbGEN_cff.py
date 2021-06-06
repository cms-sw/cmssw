import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
#from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects#, run2_GEM_2017, premix_stage1
from SimGeneral.MixingModule.mixPoolSource_cfi import *
#from SimGeneral.MixingModule.digitizers_cfi import theDigitizers

FileNames = cms.untracked.vstring(['/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/16547ECB-9C4B-E411-A815-0025905964BC.root', '/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/86C3C326-9F4B-E411-903D-0025905A48EC.root', '/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/C48D8223-9F4B-E411-BC37-0026189438DC.root', '/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/D070AB62-9D4B-E411-9766-002618FDA207.root'])

mix = cms.EDProducer("MixingModule",
    skipSignal = cms.bool(True),

    digitizers = cms.PSet(),#theDigitizers),
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

#	theMixObjects

        mixHepMC = cms.PSet(
           input = cms.VInputTag(
    		cms.InputTag("generatorSmeared","",cms.InputTag.skipCurrentProcess()),
    		cms.InputTag("generator","unsmeared",cms.InputTag.skipCurrentProcess()),
    		cms.InputTag("generator","",cms.InputTag.skipCurrentProcess())
    		),

            makeCrossingFrame = cms.untracked.bool(True),
            type = cms.string('HepMCProduct')
            )
        )
)

'''
#mix.digitizers.castor.hitsProducer = cms.InputTag("g4SimHits","CastorFI",cms.InputTag.skipCurrentProcess())
#mix.digitizers.puVtx.vtxTag = cms.InputTag("generatorSmeared","",cms.InputTag.skipCurrentProcess())
#mix.digitizers.puVtx.vtxFallbackTag = cms.InputTag("generator","",cms.InputTag.skipCurrentProcess())

mix.mixObjects.mixCH.input = cms.VInputTag(
		#cms.InputTag("g4SimHits","CaloHitsTk"), cms.InputTag("g4SimHits","CastorBU"), cms.InputTag("g4SimHits","CastorPL"), cms.InputTag("g4SimHits","CastorTU"), 
        	cms.InputTag("g4SimHits","CastorFI",cms.InputTag.skipCurrentProcess()),
        	cms.InputTag("g4SimHits","EcalHitsEB",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","EcalHitsEE",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","EcalHitsES",cms.InputTag.skipCurrentProcess()),
        	#cms.InputTag("g4SimHits","EcalTBH4BeamHits"), cms.InputTag("g4SimHits","HcalTB06BeamHits"),
        	cms.InputTag("g4SimHits","HcalHits",cms.InputTag.skipCurrentProcess()),
        	cms.InputTag("g4SimHits","ZDCHITS",cms.InputTag.skipCurrentProcess())
		)

mix.mixObjects.mixTracks.input = cms.VInputTag(
 		cms.InputTag("g4SimHits","",cms.InputTag.skipCurrentProcess())               
		)

mix.mixObjects.mixVertices.input = cms.VInputTag(
                cms.InputTag("g4SimHits","",cms.InputTag.skipCurrentProcess())
                )

mix.mixObjects.mixSH.input = cms.VInputTag(
                #cms.InputTag("g4SimHits","BSCHits"), cms.InputTag("g4SimHits","BCM1FHits"), cms.InputTag("g4SimHits","PLTHits"), cms.InputTag("g4SimHits","FP420SI"),
        	cms.InputTag("g4SimHits","MuonCSCHits",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","MuonDTHits",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","MuonRPCHits",cms.InputTag.skipCurrentProcess()),
        #cms.InputTag("g4SimHits","TotemHitsRP"), cms.InputTag("g4SimHits","TotemHitsT1"), cms.InputTag("g4SimHits","TotemHitsT2Gem"),
        	cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof",cms.InputTag.skipCurrentProcess()),
        	cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","TrackerHitsTECHighTof",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","TrackerHitsTECLowTof",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","TrackerHitsTIBHighTof",cms.InputTag.skipCurrentProcess()),
        	cms.InputTag("g4SimHits","TrackerHitsTIBLowTof",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","TrackerHitsTIDHighTof",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","TrackerHitsTIDLowTof",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","TrackerHitsTOBHighTof",cms.InputTag.skipCurrentProcess()), 
		cms.InputTag("g4SimHits","TrackerHitsTOBLowTof",cms.InputTag.skipCurrentProcess())
		)

mix.mixObjects.mixHepMC.input = cms.VInputTag(
               cms.InputTag("generatorSmeared","",cms.InputTag.skipCurrentProcess()),
               cms.InputTag("generator","unsmeared",cms.InputTag.skipCurrentProcess()),
               cms.InputTag("generator","",cms.InputTag.skipCurrentProcess())
               )

mix.mixObjects.mixHepMC.makeCrossingFrame = True

'''

mixGen = cms.Sequence(mix)
