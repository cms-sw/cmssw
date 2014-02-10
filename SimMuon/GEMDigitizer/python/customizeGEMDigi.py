import FWCore.ParameterSet.Config as cms


# PSet of mixObjects that only keeps muon collections (and SimTracks with SimVertices)
mixObjects_dt_csc_rpc =  cms.PSet(
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
            'MuonRPCHits'
        ),
        input = cms.VInputTag(
            cms.InputTag("g4SimHits","MuonCSCHits"),
            cms.InputTag("g4SimHits","MuonDTHits"),
            cms.InputTag("g4SimHits","MuonRPCHits")),
        type = cms.string('PSimHit'),
        subdets = cms.vstring(
            'MuonCSCHits',
            'MuonDTHits',
            'MuonRPCHits'
        )
    ),
    mixTracks = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits")),
        makeCrossingFrame = cms.untracked.bool(True),
        type = cms.string('SimTrack')
    )
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


# a modifier to append GEM SimHit collection definitions to mixObjects.mixSH
def mixObjects_addGEM(mixSH):
    mixSH.crossingFrames.append('MuonGEMHits')
    mixSH.input.append(cms.InputTag("g4SimHits","MuonGEMHits"))
    mixSH.subdets.append('MuonGEMHits')
    return mixSH


# customize process.mix by appending GEM SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addGEM(process):
    mixObjects_addGEM(process.mix.mixObjects.mixSH)
    return process


# Customize process.mix to be used for running muon (DT, CSC, RPC + GEM) digi only.
#  - first do such customization for (DT, CSC, RPC)
#  - append GEM SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addGEM_muon_only(process):
    process = customize_mix_muon_only(process)
    mixObjects_addGEM(process.mix.mixObjects.mixSH)
    return process


# Add simMuonGEMDigis to the list of modules served by RandomNumberGeneratorService
def customize_random_GEMDigi(process):
    process.RandomNumberGeneratorService.simMuonGEMDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    )
    return process


## load the digitizer and pad producer
def load_GEM_digitizers(process):
    process.load('SimMuon.GEMDigitizer.muonGEMDigis_cfi')
    process.load('SimMuon.GEMDigitizer.muonGEMCSCPadDigis_cfi')
    return process

# customize the full digitization sequence pdigi by adding GEMs
def customize_digi_addGEM(process):
    process = load_GEM_digitizers(process)
    process = customize_random_GEMDigi(process)
    process = customize_mix_addGEM(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis +
        process.simMuonDTDigis +
        process.simMuonRPCDigis +
        process.simMuonGEMDigis +
        process.simMuonGEMCSCPadDigis
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
    append_GEMDigi_event(process)
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
        process.simMuonGEMCSCPadDigis
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonDigi
    )
    append_GEMDigi_event(process)
    return process


# customize the digitization sequence pdigi to only digitize GEM
def customize_digi_addGEM_gem_only(process):
    process = load_GEM_digitizers(process)
    process = customize_random_GEMDigi(process)
    process = customize_mix_addGEM_muon_only(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis +
        process.simMuonDTDigis +
        process.simMuonRPCDigis +
        process.simMuonGEMDigis +
        process.simMuonGEMCSCPadDigis
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.simMuonGEMDigis*
        process.simMuonGEMCSCPadDigis
    )
    append_GEMDigi_event(process)
    return process
    
# insert the GEMDigi and GEMCSCPadDigi collection to the event
def append_GEMDigi_event(process):
    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonGEMDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simMuonGEMCSCPadDigis_*_*')
