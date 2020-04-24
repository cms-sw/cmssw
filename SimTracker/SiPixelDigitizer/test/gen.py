#
import FWCore.ParameterSet.Config as cms

process = cms.Process("GenTest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')

process.load('SimGeneral.MixingModule.mixNoPU_cfi')
 
process.load('Configuration.StandardSequences.Generator_cff')

process.load('GeneratorInterface.Core.genFilterSummary_cff')

process.load('Configuration.StandardSequences.SimIdeal_cff')
# process.load('Configuration/StandardSequences/Sim_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
# 1st file    
#        initialSeed = cms.untracked.uint32(100000000),
# 2nd file
#        initialSeed = cms.untracked.uint32(123456789),
# 3rd file    
#        initialSeed = cms.untracked.uint32(200000000),
# 4th file    
#        initialSeed = cms.untracked.uint32(300000000),
# 5th file    
#        initialSeed = cms.untracked.uint32(400000000),
# 6th file    
        initialSeed = cms.untracked.uint32(500000000),

        engineName = cms.untracked.string('HepJamesRandom')
    )
)


###########################################
# list of possible events to be generated #
###########################################
#
# single muons 
process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(100.1),
        MinPt = cms.double(99.9),
        PartID = cms.vint32(13),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single muon pt 100'),
    AddAntiParticle = cms.bool(False), # True makes to muons per event
#    firstRun = cms.untracked.uint32(1)
    firstRun = cms.untracked.uint32(2)
#    firstRun = cms.untracked.uint32(3)
#    firstRun = cms.untracked.uint32(4)
)

## # choose particle
#process.generator.PGunParameters.PartID[0] = 13
## # example: for 4 muons to test with vertex
## #process.generator.PGunParameters.PartID = cms.untracked.vint32(13,-13,13,-13)
## # example: for opposite sign back-to-back dimuon pairs set to True
## # define limits for Pt
#process.generator.PGunParameters.MinPt = 40.0
#process.generator.PGunParameters.MaxPt = 50.0
## # define limits for Pseudorapidity
#process.generator.PGunParameters.MinEta = -3
#process.generator.PGunParameters.MaxEta = 3
#process.source.firstRun = cms.untracked.uint32(10)
#process.source.firstEvent = cms.untracked.uint32(9001)
#process.source.firstLuminosityBlock = cms.untracked.uint32(10)


#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('PixelDigisTest'),
#    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('ERROR')
#    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
#)

#process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
#process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")

#process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Configuration.StandardSequences.GeometryIdeal_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')


# does using an empty PixelSkimmedGeometry.txt file speeds up job with lots more channels?
#process.siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
#    file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/longbarrel/PixelSkimmedGeometry_empty.txt')
#)
#process.es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

#process.siPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
#    file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/longbarrel/PixelSkimmedGeometry.txt')
#)
#process.es_prefer_fake_lorentz = cms.ESPrefer("SiPixelFakeLorentzAngleESSource","siPixelFakeLorentzAngleESSource")


##############################
# magnetic field in solenoid #
##############################
#
process.load('Configuration.StandardSequences.MagneticField_38T_cff')

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

#########################
# event vertex smearing #
#########################
#
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
#process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')


###########
# what is this?
# process.load("Configuration.StandardSequences.Services_cff")

# what is this?
#process.load("SimTracker.Configuration.SimTracker_cff")

# needed for global transformation
# this crashes
# process.load("Configuration.StandardSequences.FakeConditions_cff")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
#process.GlobalTag.globaltag = 'MC_53_V15::All'
#process.GlobalTag.globaltag = 'DESIGN53_V15::All'
#process.GlobalTag.globaltag = 'START53_V15::All'
# ideal
process.GlobalTag.globaltag = 'MC_70_V1::All'
# realistiv alignment and calibrations 
#process.GlobalTag.globaltag = 'START70_V1::All'

#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
#process.load("Configuration.StandardSequences.MagneticField_cff")
# include "MagneticField/Engine/data/volumeBasedMagneticField.cfi"


###############################
# global output of simulation #
###############################
#
process.o1 = cms.OutputModule(
    "PoolOutputModule",
# definition of branches to keep or drop
    outputCommands = cms.untracked.vstring('keep *','drop PCaloHits_*_*_*','drop *_*_MuonCSCHits_*',
                                           'drop *_*_MuonDTHits_*','drop *_*_MuonRPCHits_*',
                                           'drop *_*_MuonPLTHits_*','drop *_*_TotemHitsRP_*',
                                           'drop *_*_TotemHitsT1_*','drop *_*_TotemHitsT2Gem_*',
                                           ),

# definition of output file (full path)
    fileName = cms.untracked.string('/afs/cern.ch/user/d/dkotlins/work/MC/mu/pt100/simhits/simHits6.root')
)

#
process.outpath = cms.EndPath(process.o1)

#process.simulation_step = cms.Path(process.psim)
# process.digitisation_step = cms.Path(process.pdigi)

#process.p = cms.Path(process.generator*process.genParticles)
process.p = cms.Path(process.generator*process.genParticles*process.psim)

process.schedule = cms.Schedule(process.p,process.outpath)
