import FWCore.ParameterSet.Config as cms


"""
Important note to developers: the random number services of the GEM and
ME0 digi modules are no longer initialized in this file. Instead, they
are automically initialized through the --era (Run3, Phase2) command.
In case cmsRun or cmsDriver crashes because of RandomNumberGeneratorSerivce,
you may have forgotten to specify the correct era!

 --Sven Dildick
"""


# Customize process.mix to be used for running muon (DT, CSC, RPC) digi only.
#  - remove non-muon digitizers that are now run as part of mixing process
#  - delete all the digitizers' aliases.
#  - drop unnecessary mixObjects
def customize_mix_muon_only(process):
    process.mix.digitizers = digitizers = cms.PSet()
    digi_aliases = [n for n in process.aliases.keys() if 'Digi' in n]
    for a in digi_aliases: process.__delattr__(a)
    from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects
    process.mix.mixObjects = theMixObjects
    process.mix.mixObjects.mixCH = cms.PSet()
    process.mix.mixObjects.mixSH.crossingFrames = cms.untracked.vstring(
        'MuonCSCHits',
        'MuonDTHits',
        'MuonRPCHits'
    )
    process.mix.mixObjects.mixSH.input = cms.VInputTag(
        cms.InputTag("g4SimHits","MuonCSCHits"),
        cms.InputTag("g4SimHits","MuonDTHits"),
        cms.InputTag("g4SimHits","MuonRPCHits")
    )
    process.mix.mixObjects.mixSH.subdets = cms.vstring(
        'MuonCSCHits',
        'MuonDTHits',
        'MuonRPCHits'
    )
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


## load the digitizer and pad producer
def load_GEM_digitizers(process):
    process.load('SimMuon.GEMDigitizer.muonGEMDigi_cff')
    return process


## load the digitizer and pad producer
def load_ME0_digitizers(process):
    process.load('SimMuon.GEMDigitizer.muonME0Digi_cff')
    return process


# Customize process.mix to be used for running muon (DT, CSC, RPC + GEM) digi only.
#  - first do such customization for (DT, CSC, RPC)
#  - append GEM SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addGEM_muon_only(process):
    process = customize_mix_muon_only(process)
    process = customize_mix_addGEM(process)
    return process


# Customize process.mix to be used for running muon (DT, CSC, RPC + ME0) digi only.
#  - first do such customization for (DT, CSC, RPC)
#  - append ME0 SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addME0_muon_only(process):
    process = customize_mix_muon_only(process)
    process = customize_mix_addME0(process)
    return process


# Customize process.mix to be used for running muon (DT, CSC, RPC, GEM, ME0) digi only.
#  - first do such customization for (DT, CSC, RPC)
#  - append GEM SimHit collection definitions to mix.mixObjects.mixSH
#  - append ME0 SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addGEM_addME0_muon_only(process):
    process = customize_mix_muon_only(process)
    process = customize_mix_addGEM(process)
    process = customize_mix_addME0(process)
    return process


# customize the digitization sequence pdigi to only digitize GEM
def customize_digi_addGEM_gem_only(process):
    process = load_GEM_digitizers(process)
    process = customize_mix_addGEM_muon_only(process)
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonGEMDigi
    )
    process = append_GEMDigi_event(process)
    return process


# customize the digitization sequence pdigi to only digitize ME0
def customize_digi_addME0_me0_only(process):
    process = load_ME0_digitizers(process)
    process = customize_mix_addME0_muon_only(process)
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonME0Digi
    )
    process = append_ME0Digi_event(process)
    return process


# customize the digitization sequence pdigi to only digitize GEM, ME0
def customize_digi_addGEM_addME0_gem_me0_only(process):
    process = load_GEM_digitizers(process)
    process = load_ME0_digitizers(process)
    process = customize_mix_addGEM_addME0_muon_only(process)
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonGEMDigi*
        process.muonME0Digi
    )
    process = append_GEMDigi_event(process)
    return process


# customize the digitization sequence pdigi to only digitize DT+CSC+RPC+GEM
def customize_digi_addGEM_muon_only(process):
    process = load_GEM_digitizers(process)
    process = customize_mix_addGEM_muon_only(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis +
        process.simMuonDTDigis +
        process.simMuonRPCDigis +
        process.muonGEMDigi
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonDigi
    )
    process = append_GEMDigi_event(process)
    return process


# customize the digitization sequence pdigi to only digitize DT+CSC+RPC+GEM+ME0
def customize_digi_addGEM_addME0_muon_only(process):
    process = load_GEM_digitizers(process)
    process = load_ME0_digitizers(process)
    process = customize_mix_addGEM_addME0_muon_only(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis +
        process.simMuonDTDigis +
        process.simMuonRPCDigis +
        process.muonGEMDigi +
        process.muonME0Digi
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonDigi
    )
    process = append_GEMDigi_event(process)
    return process


# customize the full digitization sequence pdigi by adding GEMs
def customize_digi_addGEM(process):
    process = load_GEM_digitizers(process)
    process = customize_mix_addGEM(process)
    process.doAllDigi = cms.Sequence(
        process.calDigi +
        process.muonDigi +
        process.muonGEMDigi
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.doAllDigi*
        process.addPileupInfo
    )
    process = append_GEMDigi_event(process)
    return process


# customize the full digitization sequence pdigi by adding GEMs
def customize_digi_addGEM_addME0(process):
    process = load_GEM_digitizers(process)
    process = load_ME0_digitizers(process)
    process = customize_mix_addGEM(process)
    process = customize_mix_addME0(process)
    process.doAllDigi = cms.Sequence(
        process.calDigi +
        process.muonDigi +
        process.muonGEMDigi +
        process.muonME0Digi
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.doAllDigi*
        process.addPileupInfo
    )
    process = append_GEMDigi_event(process)
    return process


# insert the GEMDigi and GEMPadDigi collection to the event
def append_GEMDigi_event(process):
    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.extend(['keep *_g4SimHits_Muon*_*', 'keep *_*Muon*_*_*'])

    return process

# Customizations for the background
def customize_digi_noGEMbkg(process):
    process.simMuonGEMDigis.doBkgNoise = False
    return process

def customize_digi_noGEMsafety(process):
    process.simMuonGEMDigis.rateFact = 1
    return process

# Customizations for the background (to be updated once the realistic digi is in place)
def customize_digi_noME0bkg(process):
    process.simMuonME0PseudoDigis.simulateElectronBkg = False
    process.simMuonME0PseudoDigis.simulateNeutralBkg = False
    return process

def customize_digi_noME0safety(process):
    process.simMuonME0PseudoDigis.rateFact = 1
    return process

# Customizations for the background
def customize_digi_noRPCbkg(process):
    process.simMuonRPCDigis.doBkgNoise = False
    return process

# adding re-digi costumisation - to be used for dedicated trigger studies
def customise_rpcRedigi(process):
    process.load('Configuration.StandardSequences.Digi_cff')
    process.simMuonRPCReDigis = process.simMuonRPCDigis.clone()
    process.simMuonRPCReDigis.digiModelConfig = process.simMuonRPCDigis.digiModelConfig.clone(
        IRPC_time_resolution = cms.double(1.5),
        IRPC_electronics_jitter = cms.double(0.1),
    )
    process.simMuonRPCReDigis.digiIRPCModelConfig = process.simMuonRPCReDigis.digiModelConfig.clone(
        IRPC_time_resolution = cms.double(1.0),
        do_Y_coordinate = cms.bool(True),
    )
    process.simMuonRPCReDigis.digiModel = cms.string('RPCSimModelTiming')
    process.RandomNumberGeneratorService.simMuonRPCReDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(13579),
        engineName = cms.untracked.string('TRandom3')
    )
    process.rpcRecHits.rpcDigiLabel = cms.InputTag("simMuonRPCReDigis")
    process.validationMuonRPCDigis.rpcDigiTag = cms.untracked.InputTag("simMuonRPCReDigis")
    process.reconstruction_step.replace(
        process.rpcRecHits,
        cms.Sequence(process.simMuonRPCReDigis+process.rpcRecHits)
    )
    return process
