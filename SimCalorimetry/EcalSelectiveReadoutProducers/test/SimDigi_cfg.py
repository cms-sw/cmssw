import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")



# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

#process.load('Configuration.StandardSequences.SimIdeal_cff')
#process.load('Configuration.StandardSequences.Digi_cff')


#Conditions:
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_38Y_V1::All'

# event vertex smearing - applies only once (internal check)
# Note : all internal generators will always do (0,0,0) vertex
#
process.load('Configuration.StandardSequences.VtxSmearedEarly10TeVCollision_cff')


#Geometry
#
process.load('Configuration.StandardSequences.GeometryExtended_cff')

# Calo geometry service model
#process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

#process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

#process.load("Geometry.EcalMapping.EcalMapping_cfi")

#process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

#Magnetic Field
#
#process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')

# Step 1: Event generator
process.load('Configuration.StandardSequences.Generator_cff')
process.generator = cms.EDProducer("FlatRandomEGunProducer",
  PGunParameters = cms.PSet(
  # you can request more than 1 particle
     PartID = cms.vint32(11),
     MaxEta = cms.double(3.0),
     MaxPhi = cms.double(3.141592653589793),
     MinEta = cms.double(-3.0),
     MinE = cms.double(99.99),
     MinPhi = cms.double(-3.141592653589793), ## must be in radians

     MaxE = cms.double(100.01)
  ),
  firstRun = cms.untracked.uint32(1),
  AddAntiParticle = cms.bool(True),
  Verbosity = cms.untracked.int32(0) ## set to 1 (or greater)  for printouts                                   
)


# Step 2 : CMS Detector Simulation
#process.load('Configuration.StandardSequences.Sim_cff')
process.load("SimG4Core.Application.g4SimHits_cfi")

# Pile-up processing:
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

#
# Ecal digi production:
process.load("SimCalorimetry.EcalSimProducers.ecaldigi_cfi")

# TPG: defines the ecalTriggerPrimitiveDigis module
process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.o1 = cms.OutputModule("PoolOutputModule",
    compressionLevel = cms.untracked.int32(9),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_simEcalUnsuppressedDigis_*_*', 
        'keep *_simEcalUnsuppressedDigis_*_*', 
        'keep *_simEcalTriggerPrimitiveDigis_*_*'),
    fileName = cms.untracked.string('srp_validation_in.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
)

process.RandomNumberGeneratorService= cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        VtxSmeared = cms.untracked.uint32(123456781),
        g4SimHits = cms.untracked.uint32(123456782),
        ecalUnsuppressedDigis = cms.untracked.uint32(123456783),
        simEcalUnsuppressedDigis = cms.untracked.uint32(123456784),
        generator = cms.untracked.uint32(123456785)
    )
)

process.source = cms.Source("EmptySource")

#process.load("Configuration.GenProduction.PythiaUESettings_cfi")

process.detSim = cms.Sequence(process.VtxSmeared*process.g4SimHits)
process.p = cms.Path(process.generator*process.detSim*process.mix*process.simEcalUnsuppressedDigis*process.simEcalTriggerPrimitiveDigis)
process.outpath = cms.EndPath(process.o1)

