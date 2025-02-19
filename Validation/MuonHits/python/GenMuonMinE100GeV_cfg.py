import FWCore.ParameterSet.Config as cms

process = cms.Process("Gen")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('FwkJob'),
    destinations = cms.untracked.vstring('cout')
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.untracked.uint32(98765)
)

process.source = cms.Source("FlatRandomEGunSource",
    maxEvents = cms.untracked.int32(100),
    PGunParameters = cms.untracked.PSet(
        # you can request more than 1 particle
        #untracked vint32  PartID = {211,11}
        PartID = cms.untracked.vint32(13),
        MaxEta = cms.untracked.double(2.4),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-2.4),
        MinE = cms.untracked.double(100.0),
        MinPhi = cms.untracked.double(-3.14159265359), ## in radians

        MaxE = cms.untracked.double(100.0)
    ),
    Verbosity = cms.untracked.int32(0) ## set to 1 (or greater)  for printouts

)

process.GEN = cms.OutputModule("PoolOutputModule",
    datasets = cms.untracked.PSet(
        dataset1 = cms.untracked.PSet(
            dataTier = cms.untracked.string('GEN')
        )
    ),
    fileName = cms.untracked.string('mu_minus_e100GeV.root')
)

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Validation/MuonHits/python/GenMuonMinE100GeV_cfg.py,v $'),
    annotation = cms.untracked.string('gen. muons for muon subsystems validation scan')
)
process.outpath = cms.EndPath(process.GEN)

