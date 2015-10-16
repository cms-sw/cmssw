import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.aliases_cfi import * 
# from SimGeneral.MixingModule.mixObjects_cfi import *
# from SimGeneral.MixingModule.digitizers_cfi import *
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
# from SimGeneral.MixingModule.trackingTruthProducer_cfi import *

# PSet of mixObjects that only keeps muon collections (and SimTracks with SimVertices)
mixObjects_dt_csc_rpc =  cms.PSet(
    mixCH = cms.PSet(
        crossingFrames = cms.untracked.vstring(),
        input = cms.VInputTag(),
        type = cms.string('PCaloHit'),
        subdets = cms.vstring()
    ),
    mixHepMC = cms.PSet(
        input = cms.VInputTag(cms.InputTag("generatorSmeared"),cms.InputTag("generator")),
        makeCrossingFrame = cms.untracked.bool(True),
        type = cms.string('HepMCProduct')
    ),
    mixVertices = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits")),
        makeCrossingFrame = cms.untracked.bool(True),
        type = cms.string('SimVertex')
    ),
    mixSH = cms.PSet(
        crossingFrames = cms.untracked.vstring(
            'MuonCSCHits',
            'MuonDTHits',
            'MuonRPCHits',
            'TrackerHitsPixelBarrelHighTof',
            'TrackerHitsPixelBarrelLowTof',
            'TrackerHitsPixelEndcapHighTof',
            'TrackerHitsPixelEndcapLowTof',
            'TrackerHitsTECHighTof',
            'TrackerHitsTECLowTof',
            'TrackerHitsTIBHighTof',
            'TrackerHitsTIBLowTof',
            'TrackerHitsTIDHighTof',
            'TrackerHitsTIDLowTof',
            'TrackerHitsTOBHighTof',
            'TrackerHitsTOBLowTof'
            ),
        input = cms.VInputTag(
            cms.InputTag("g4SimHits","MuonCSCHits"),
            cms.InputTag("g4SimHits","MuonDTHits"),
            cms.InputTag("g4SimHits","MuonRPCHits"),
            cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),     
            cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTECHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTECLowTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"),     
            ),
        type = cms.string('PSimHit'),
        subdets = cms.vstring(
            'MuonCSCHits',
            'MuonDTHits',
            'MuonRPCHits',
            'TrackerHitsPixelBarrelHighTof',
            'TrackerHitsPixelBarrelLowTof',
            'TrackerHitsPixelEndcapHighTof',
            'TrackerHitsPixelEndcapLowTof',
            'TrackerHitsTECHighTof',
            'TrackerHitsTECLowTof',
            'TrackerHitsTIBHighTof',
            'TrackerHitsTIBLowTof',
            'TrackerHitsTIDHighTof',
            'TrackerHitsTIDLowTof',
            'TrackerHitsTOBHighTof',
            'TrackerHitsTOBLowTof'
            )
    ),
    mixTracks = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits")),
        makeCrossingFrame = cms.untracked.bool(True),
        type = cms.string('SimTrack')
    )
)

# PSet of mixObjects that only keep muon and tracker collections (and SimTracks with SimVertices)
mixObjects_dt_csc_rpc_trk =  cms.PSet(
    mixCH = cms.PSet(
        crossingFrames = cms.untracked.vstring(),
        input = cms.VInputTag(),
        type = cms.string('PCaloHit'),
        subdets = cms.vstring()
    ),
    mixHepMC = cms.PSet(
        input = cms.VInputTag(cms.InputTag("generator")),
        makeCrossingFrame = cms.untracked.bool(True),
        type = cms.string('HepMCProduct')
    ),
    mixVertices = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits")),
        makeCrossingFrame = cms.untracked.bool(True),
        type = cms.string('SimVertex')
    ),
    mixSH = cms.PSet(
        crossingFrames = cms.untracked.vstring(
            'MuonCSCHits',
            'MuonDTHits',
            'MuonRPCHits',
            'TrackerHitsPixelBarrelHighTof',
            'TrackerHitsPixelBarrelLowTof',
            'TrackerHitsPixelEndcapHighTof',
            'TrackerHitsPixelEndcapLowTof',
            'TrackerHitsTECHighTof',
            'TrackerHitsTECLowTof',
            'TrackerHitsTIBHighTof',
            'TrackerHitsTIBLowTof',
            'TrackerHitsTIDHighTof',
            'TrackerHitsTIDLowTof',
            'TrackerHitsTOBHighTof',
            'TrackerHitsTOBLowTof'
            ),
        input = cms.VInputTag(
            cms.InputTag("g4SimHits","MuonCSCHits"),
            cms.InputTag("g4SimHits","MuonDTHits"),
            cms.InputTag("g4SimHits","MuonRPCHits"),
            cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),     
            cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTECHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTECLowTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"),     
            cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"),     
            ),
        type = cms.string('PSimHit'),
        subdets = cms.vstring(
            'MuonCSCHits',
            'MuonDTHits',
            'MuonRPCHits',
            'TrackerHitsPixelBarrelHighTof',
            'TrackerHitsPixelBarrelLowTof',
            'TrackerHitsPixelEndcapHighTof',
            'TrackerHitsPixelEndcapLowTof',
            'TrackerHitsTECHighTof',
            'TrackerHitsTECLowTof',
            'TrackerHitsTIBHighTof',
            'TrackerHitsTIBLowTof',
            'TrackerHitsTIDHighTof',
            'TrackerHitsTIDLowTof',
            'TrackerHitsTOBHighTof',
            'TrackerHitsTOBLowTof'
            )
    ),
    mixTracks = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits")),
        makeCrossingFrame = cms.untracked.bool(True),
        type = cms.string('SimTrack')
    ),
)


# Customize process.mix to be used for running muon (DT, CSC, RPC) digi only.
#  - remove non-muon digitizers that are now run as part of mixing process
#  - delete all the digitizers' aliases.
#  - drop unnecessary mixObjects
def customize_mix_muon_only(process):
    process.mix.digitizers = digitizers = cms.PSet()
    digi_aliases = filter(lambda n: 'Digi' in n, process.aliases.keys())
    for a in digi_aliases: process.__delattr__(a)
    process.mix.mixObjects = mixObjects_dt_csc_rpc
    return process

# Customize process.mix to be used for running muon and tracker digi only.
#  - remove calo digitizers that are now run as part of mixing process
#  - delete all the digitizers' aliases apart of pixel and strip aliasses.
#  - reset the simCastorDigis, simEcalUnsuppressedDigis 
#          and simHcalUnsuppressedDigis
#  - drop unnecessary mixObjects
def customize_mix_nocalo(process):
    process.mix.digitizers = digitizers = cms.PSet(
          pixel = cms.PSet(
          pixelDigitizer
       ),
       strip = cms.PSet(
           stripDigitizer
       ),
    )
    process.mix.theDigitizersValid = cms.PSet(
        pixel = cms.PSet(
            pixelDigitizer
            ),
        strip = cms.PSet(
            stripDigitizer
            )
    )
    # delete some contents of SimGeneral/MixingModule/python/aliases_cfi.py
    # i was not able to delete these processes in a different way
    process.simCastorDigis = cms.EDAlias()
    process.simEcalUnsuppressedDigis = cms.EDAlias()
    process.simHcalUnsuppressedDigis = cms.EDAlias()
    # delete all digitizer aliasses apart of pixel and strip aliases
    digi_aliases = filter(lambda n: 'Digi' in n, process.aliases.keys())
    print("digi aliases before clean up: ")
    for a in digi_aliases: 
        print(a)
    if ('Strip' not in a) and ('Pixel' not in a): 
            process.__delattr__(a)
    process.mix.mixObjects = mixObjects_dt_csc_rpc_trk
    return process

# Customize process.mix to be used for running muon (DT, CSC, RPC + GEM) digi only.
#  - first do such customization for (DT, CSC, RPC)
#  - append GEM SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addGEM_muon_only(process):
    process = customize_mix_muon_only(process)
    process = customize_mix_addGEM(process)
    return process

def customize_mix_addGEM_nocalo(process):
    process = customize_mix_nocalo(process)
    process = customize_mix_addGEM(process)
    return process

# customize the digitization sequence pdigi to only digitize DT+CSC+RPC+GEM
def customize_digi_addGEM_muon_only(process):
    process = load_GEM_digitizers(process)
    process = customize_random_GEMDigi(process)
    process = customize_mix_addGEM_muon_only(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis + 
        process.simMuonDTDigis + 
        process.simMuonRPCDigis + 
        process.simMuonGEMDigis + 
        process.simMuonGEMPadDigis
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonDigi
    )
    process = append_GEMDigi_event(process)
    return process

# customize the digitization sequence pdigi to only digitize DT+CSC+RPC+GEM
def customize_digi_addGEM_nocalo(process):
    process = load_GEM_digitizers(process)
    process = customize_random_GEMDigi(process)
    process = customize_mix_addGEM_nocalo(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis + 
        process.simMuonDTDigis + 
        process.simMuonRPCDigis + 
        process.simMuonGEMDigis + 
        process.simMuonGEMPadDigis
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonDigi
    )
    # process.pdigi.remove(simCastorDigis)
    process = append_GEMDigi_event(process)
    return process





# customize process.mix by appending GEM SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addGEM(process):
    process.mix.mixObjects.mixSH.crossingFrames.append('MuonGEMHits')
    process.mix.mixObjects.mixSH.input.append(cms.InputTag("g4SimHits","MuonGEMHits"))
    process.mix.mixObjects.mixSH.subdets.append('MuonGEMHits')
    return process


# customize process.mix by appending ME0 SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addME0(process):
    process.mix.mixObjects.mixSH.crossingFrames.append('MuonME0Hits')
    process.mix.mixObjects.mixSH.input.append(cms.InputTag("g4SimHits","MuonME0Hits"))
    process.mix.mixObjects.mixSH.subdets.append('MuonME0Hits')
    return process


# Customize process.mix to be used for running muon (DT, CSC, RPC + GEM) digi only.
#  - first do such customization for (DT, CSC, RPC)
#  - append GEM SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addGEM_addME0_muon_only(process):
    process = customize_mix_muon_only(process)
    process = customize_mix_addGEM(process)
    process = customize_mix_addME0(process)
    return process

def customize_mix_addGEM_addME0_nocalo(process):
    process = customize_mix_nocalo(process)
    process = customize_mix_addGEM(process)
    process = customize_mix_addME0(process)
    return process


# Add simMuonGEMDigis to the list of modules served by RandomNumberGeneratorService
def customize_random_GEMDigi(process):
    process.RandomNumberGeneratorService.simMuonGEMDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    )
    return process


# Add simMuonME0Digis to the list of modules served by RandomNumberGeneratorService
def customize_random_ME0Digi(process):
    process.RandomNumberGeneratorService.simMuonME0Digis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    )
    return process


## load the digitizer and pad producer
def load_GEM_digitizers(process):
    process.load('SimMuon.GEMDigitizer.muonGEMDigi_cff')
    return process


## load the digitizer and pad producer
def load_ME0_digitizers(process):
    process.load('SimMuon.GEMDigitizer.muonME0DigisPreReco_cfi')
    return process


# customize the full digitization sequence pdigi by adding GEMs
def customize_digi_addGEM(process):
    process = load_GEM_digitizers(process)
    process = customize_random_GEMDigi(process)
    process = customize_mix_addGEM(process)
    process.doAllDigi = cms.Sequence(
        process.calDigi + 
        process.muonDigi
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.doAllDigi*
        process.addPileupInfo
    )
    process = append_GEMDigi_event(process)
    return process


# customize the digitization sequence pdigi to only digitize GEM
def customize_digi_addGEM_gem_only(process):
    process = load_GEM_digitizers(process)
    process = customize_random_GEMDigi(process)
    process = customize_mix_addGEM_muon_only(process)
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.simMuonGEMDigis*
        process.simMuonGEMPadDigis 
    )
    process = append_GEMDigi_event(process)
    return process

    
# customize the full digitization sequence pdigi by adding GEMs
def customize_digi_addGEM_addME0(process):
    process = load_GEM_digitizers(process)
    process = load_ME0_digitizers(process)
    process = customize_random_GEMDigi(process)
    process = customize_random_ME0Digi(process)
    process = customize_mix_addGEM(process)
    process = customize_mix_addME0(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis +
        process.simMuonDTDigis +
        process.simMuonRPCDigis +
        process.simMuonGEMDigis +
        process.simMuonGEMPadDigis +
        process.simMuonME0Digis
    )
    process.doAllDigi = cms.Sequence(
        process.calDigi +
        process.muonDigi
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.doAllDigi*
        process.addPileupInfo
    )
    process = append_GEMDigi_event(process)
    return process


# customize the digitization sequence pdigi to only digitize DT+CSC+RPC+GEM
def customize_digi_addGEM_addME0_muon_only(process):
    process = load_GEM_digitizers(process)
    process = load_ME0_digitizers(process)
    process = customize_random_GEMDigi(process)
    process = customize_random_ME0Digi(process)
    process = customize_mix_addGEM_addME0_muon_only(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis +
        process.simMuonDTDigis +
        process.simMuonRPCDigis +
        process.simMuonGEMDigis +
        process.simMuonGEMPadDigis +
        process.simMuonME0Digis
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonDigi
    )
    process = append_GEMDigi_event(process)
    return process


# customize the digitization sequence pdigi to only digitize GEM
def customize_digi_addGEM_addME0_gem_only(process):
    process = load_GEM_digitizers(process)
    process = load_ME0_digitizers(process)
    process = customize_random_GEMDigi(process)
    process = customize_random_ME0Digi(process)
    process = customize_mix_addGEM_addME0_muon_only(process)
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.simMuonGEMDigis*
        process.simMuonGEMPadDigis*
        process.simMuonME0Digis
    )
    process = append_GEMDigi_event(process)
    return process


# insert the GEMDigi and GEMPadDigi collection to the event
def append_GEMDigi_event(process):
    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('drop *')
            # getattr(process,b).outputCommands.append('keep *_mix_*_*')
            getattr(process,b).outputCommands.append('keep *_g4SimHits__*')
            getattr(process,b).outputCommands.append('keep *_g4SimHits_Muon*_*')
            getattr(process,b).outputCommands.append('keep *_g4SimHits_Tracker*_*')
            getattr(process,b).outputCommands.append('keep *_generator_*_*')
            getattr(process,b).outputCommands.append('keep *_genParticles_*_*')
            getattr(process,b).outputCommands.append('keep *_*Muon*_*_*')
            getattr(process,b).outputCommands.append('keep *_*Strip*_*_*')
            getattr(process,b).outputCommands.append('keep *_*Pixel*_*_*')
            # getattr(process,b).outputCommands.append('keep *_*_*_*')
            # getattr(process,b).outputCommands.append('keep *_simSiPixelDigis_*_*')
            # getattr(process,b).outputCommands.append('keep *_simSiStripDigis_*_*')

    return process
