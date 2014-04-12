import FWCore.ParameterSet.Config as cms

process = cms.Process("Gen")
# setup useful services
#
process.load("Validation.GlobalHits.Random_cfi")

process.load("Validation.GlobalHits.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Pythia settings for ttbar generation
#
process.load("Configuration.Generator.PythiaUESettings_cfi")

# smearing of the MC vertex
#
#module VtxSmeared
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.source = cms.Source("PythiaSource",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    #untracked int32 maxEvents = 10000
    pythiaPylistVerbosity = cms.untracked.int32(0),
    pythiaVerbosity = cms.untracked.bool(False),
    PythiaParameters = cms.PSet(
        process.pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0                  ! User defined processes', 
            'MSUB(81) = 1            ! qqbar to QQbar', 
            'MSUB(82) = 1            ! gg to QQbar', 
            'MSTP(7) = 6             ! flavour = top', 
            'PMAS(6,1) = 175.        ! top quark mass'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)

process.GEN = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN')
    ),
    fileName = cms.untracked.string('MC.root')
)

process.p1 = cms.Path(process.VtxSmeared)
process.outpath = cms.EndPath(process.GEN)

