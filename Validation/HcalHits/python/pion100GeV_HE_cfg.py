import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")
# this will run plig-in energy-flat random particle gun
# and puts particles (HepMCPRoduct) into edm::Event
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.untracked.uint32(54321)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        # you can request more than 1 particle
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(2.108),
        MaxPhi = cms.untracked.double(0.6109),
        MinEta = cms.untracked.double(2.108),
        MinE = cms.untracked.double(100.0),
        MinPhi = cms.untracked.double(0.6109), ## it must be in radians

        MaxE = cms.untracked.double(100.0)
    ),
    Verbosity = cms.untracked.int32(0), ## for printouts, set it to 1 (or greater)

    psethack = cms.string('single pion 100GeV on endcap'),
    firstRun = cms.untracked.uint32(1)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('mc_pi+100_etaphi244.root')
)

process.p = cms.EndPath(process.GEN)

