import FWCore.ParameterSet.Config as cms

process = cms.Process("TestProcess")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.writer = cms.EDFilter("FlatEGunASCIIWriter",
    # you can give it or not ; the default is FlatEGunHepMC.dat"
    OutFileName = cms.untracked.string('single_neutrino.random.dat'),
    PGunParameters = cms.PSet(
        PartID = cms.vint32(14),
        MaxEta = cms.double(5.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-5.5),
        MinE = cms.double(9.99),
        MinPhi = cms.double(-3.14159265359),
        MaxE = cms.double(10.01)
    )
)

process.p1 = cms.Path(process.writer)

