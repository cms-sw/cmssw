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


# a modifier to append ME0 SimHit collection definitions to mixObjects.mixSH
def mixObjects_addME0(mixSH):
    mixSH.crossingFrames.append('MuonME0Hits')
    mixSH.input.append(cms.InputTag("g4SimHits","MuonME0Hits"))
    mixSH.subdets.append('MuonME0Hits')
    return mixSH


# customize process.mix by appending ME0 SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addME0(process):
    mixObjects_addME0(process.mix.mixObjects.mixSH)
    return process


# Customize process.mix to be used for running muon (DT, CSC, RPC + ME0) digi only.
#  - first do such customization for (DT, CSC, RPC)
#  - append ME0 SimHit collection definitions to mix.mixObjects.mixSH
def customize_mix_addME0_muon_only(process):
    process = customize_mix_muon_only(process)
    mixObjects_addME0(process.mix.mixObjects.mixSH)
    return process


# Add simMuonME0Digis to the list of modules served by RandomNumberGeneratorService
def customize_random_ME0Digi(process):
    process.RandomNumberGeneratorService.simMuonME0Digis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    )
    return process


# customize the full digitization sequence pdigi by adding ME0s
def customize_digi_addME0(process):
    process = customize_random_ME0Digi(process)
    process = customize_mix_addME0(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis +
        process.simMuonDTDigis +
        process.simMuonRPCDigis +
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
        process.trackingParticles*
        process.addPileupInfo
    )
    return process


# customize the digitization sequence pdigi to only digitize DT+CSC+RPC+ME0
def customize_digi_addME0_muon_only(process):
    process = customize_random_ME0Digi(process)
    process = customize_mix_addME0_muon_only(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis +
        process.simMuonDTDigis +
        process.simMuonRPCDigis +
        process.simMuonME0Digis
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonDigi
    )
    return process


# customize the digitization sequence pdigi to only digitize ME0
def customize_digi_addME0_me0_only(process):
    process = customize_random_ME0Digi(process)
    process = customize_mix_addME0_muon_only(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis +
        process.simMuonDTDigis +
        process.simMuonRPCDigis +
        process.simMuonME0Digis
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.simMuonME0Digis
    )
    return process

