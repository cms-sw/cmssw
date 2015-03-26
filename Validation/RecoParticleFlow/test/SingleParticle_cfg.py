import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

#generation
# Flat energy gun
process.source = cms.Source(
    "FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(==PID==),
        MinEta = cms.untracked.double(-5.00000),
        MaxEta = cms.untracked.double(5.00000),
        MinPhi = cms.untracked.double(-3.14159), ## in radians
        MaxPhi = cms.untracked.double(3.14159),
        MinE = cms.untracked.double(==EMIN==),
        MaxE = cms.untracked.double(==EMAX==),
    ),
    Verbosity = cms.untracked.int32(0) ## set to 1 (or greater)  for printouts
)
"""
# Flat pT gun
process.source = cms.Source(
    "FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(==PID==),
        MinEta = cms.untracked.double(-5.10000),
        MaxEta = cms.untracked.double(5.10000),
        MinPhi = cms.untracked.double(-3.14159), ## in radians
        MaxPhi = cms.untracked.double(3.14159),
        MinPt = cms.untracked.double(==PTMIN==),
        MaxPt = cms.untracked.double(==PTMAX==),
    ),
    Verbosity = cms.untracked.int32(0) ## set to 1 (or greater)  for printouts
)
"""

# this example configuration offers some minimum 
# annotation, to help users get through; please
# don't hesitate to read through the comments
# use MessageLogger to redirect/suppress multiple
# service messages coming from the system
#
# in this config below, we use the replace option to make
# the logger let out messages of severity ERROR (INFO level
# will be suppressed), and we want to limit the number to 10
#
process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.MagneticField_40T_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_30X::All"

process.load("FWCore.MessageService.MessageLogger_cfi")

# this config frament brings you the generator information
process.load("Configuration.StandardSequences.Generator_cff")

# this config frament brings you 3 steps of the detector simulation:
# -- vertex smearing (IR modeling)
# -- G4-based hit level detector simulation
# -- digitization (electronics readout modeling)
# it returns 2 sequences : 
# -- psim (vtx smearing + G4 sim)
# -- pdigi (digitization in all subsystems, i.e. tracker=pix+sistrips,
#           cal=ecal+ecal-0-suppression+hcal), muon=csc+dt+rpc)
#
process.load("Configuration.StandardSequences.Simulation_cff")

process.RandomNumberGeneratorService.theSource.initialSeed= ==SEED==
#process.RandomNumberGeneratorService.theSource.initialSeed= 1414

# please note the IMPORTANT: 
# in order to operate Digis, one needs to include Mixing module 
# (pileup modeling), at least in the 0-pileup mode
#
# There're 3 possible configurations of the Mixing module :
# no-pileup, low luminosity pileup, and high luminosity pileup
#
# they come, respectively, through the 3 config fragments below
#
# *each* config returns label "mix"; thus you canNOT have them
# all together in the same configuration, but only one !!!
#
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

#include "Configuration/StandardSequences/data/MixingLowLumiPileUp.cff" 
#include "Configuration/StandardSequences/data/MixingHighLumiPileUp.cff" 
process.load("Configuration.StandardSequences.L1Emulator_cff")

process.load("Configuration.StandardSequences.DigiToRaw_cff")

process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("Configuration.StandardSequences.VtxSmearedEarly10TeVCollision_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.MessageLogger = cms.Service("MessageLogger",
    reco = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
   destinations = cms.untracked.vstring('reco')
)

process.aod = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('aod.root')
)

process.reco = cms.OutputModule("PoolOutputModule",
    process.RECOSIMEventContent,
    fileName = cms.untracked.string('reco.root')
)

process.load("RecoParticleFlow.Configuration.Display_EventContent_cff")
process.display = cms.OutputModule("PoolOutputModule",
    process.DisplayEventContent,
    fileName = cms.untracked.string('==OUTPUT==')
)

process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")

#process.electronChi2.MaxChi2 = 100000
#process.electronChi2.nSigma = 3
#process.pfTrackElec.ModeMomentum = True
#process.pfTrackElec.AddGSFTkColl = False
#process.particleFlowBlock.pf_chi2_ECAL_Track = 900

process.p0 = cms.Path(process.pgen)
process.p1 = cms.Path(process.psim)
process.p2 = cms.Path(process.pdigi)
process.p3 = cms.Path(process.L1Emulator)
process.p4 = cms.Path(process.DigiToRaw)
process.p5= cms.Path(process.RawToDigi)
process.p6= cms.Path(process.reconstruction+process.particleFlowSimParticle)
#process.outpath = cms.EndPath(process.aod+process.reco+process.display)
process.outpath = cms.EndPath(process.display)
process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.p3,process.p4,process.p5,process.p6,process.outpath)



