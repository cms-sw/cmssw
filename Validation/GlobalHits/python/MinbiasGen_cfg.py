import FWCore.ParameterSet.Config as cms

process = cms.Process("Gen")
# setup useful services
#
process.load("Validation.GlobalHits.Random_cfi")

process.load("Validation.GlobalHits.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# min bias Pythia
process.load("GeneratorInterface.Pythia6Interface.PythiaSourceMinBias_cfi")

# smearing of the MC vertex
#
#module VtxSmeared
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.GEN = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN')
    ),
    fileName = cms.untracked.string('MC.root')
)

process.p1 = cms.Path(process.VtxSmeared)
process.outpath = cms.EndPath(process.GEN)
process.PythiaSource.pythiaHepMCVerbosity = False
process.PythiaSource.pythiaPylistVerbosity = 0

