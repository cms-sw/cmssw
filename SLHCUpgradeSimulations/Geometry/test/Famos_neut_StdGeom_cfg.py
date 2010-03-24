import FWCore.ParameterSet.Config as cms

process = cms.Process("FastsimNeutRad")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.RandomNumberGeneratorService.simSiStripDigis = cms.PSet(
      initialSeed = cms.untracked.uint32(1234567),
      engineName = cms.untracked.string('HepJamesRandom'))
process.RandomNumberGeneratorService.simSiPixelDigis = cms.PSet(
      initialSeed = cms.untracked.uint32(1234567),
      engineName = cms.untracked.string('HepJamesRandom'))

process.load("FastSimulation/Configuration/FlatPtMuonGun_cfi")
process.generator.PGunParameters.PartID[0] = 14
process.generator.PGunParameters.MinPt = 35.0
process.generator.PGunParameters.MaxPt = 50.0
process.generator.PGunParameters.MinEta = -4.0
process.generator.PGunParameters.MaxEta = 4.0
process.generator.AddAntiParticle = False

# from std full sim 
#process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V8::All'

# Famos sequences
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
## replace with strawman geometry
#process.load("SLHCUpgradeSimulations.Geometry.PhaseI_cmsSimIdealGeometryXML_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

process.NeutRad = cms.EDAnalyzer("NeutRadtuple",
                                 OutputFile = cms.string("neutrad_std_famos_ntuple.root")
                                 )

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

process.Timing =  cms.Service("Timing")
process.load("FWCore/MessageService/MessageLogger_cfi")
#process.MessageLogger.destinations = cms.untracked.vstring("detailedInfo_phase1_neutrad")

process.famosSimHits.VertexGenerator.type = 'None'

# Just Famos
process.p0 = cms.Path(process.generator)
#process.p1 = cms.Path(process.offlineBeamSpot)
process.p2 = cms.Path(process.famosSimHits)
process.p9 = cms.Path(process.NeutRad)
#process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.p9)
process.schedule = cms.Schedule(process.p0,process.p2,process.p9)

