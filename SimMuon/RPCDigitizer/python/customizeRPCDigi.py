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

# customize the digitization sequence pdigi to only digitize DT+CSC+RPC
def customize_digi_muon_only(process):
    process = customize_mix_muon_only(process)
    process.muonDigi = cms.Sequence(
        process.simMuonCSCDigis + 
        process.simMuonDTDigis + 
        process.simMuonRPCDigis
    )
    process.pdigi = cms.Sequence(
        cms.SequencePlaceholder("randomEngineStateProducer")*
        cms.SequencePlaceholder("mix")*
        process.muonDigi
    )
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
