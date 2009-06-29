import FWCore.ParameterSet.Config as cms


process = cms.Process("PROD1")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)

#generation
process.load("RecoParticleFlow.Configuration.source_singleTau_cfi")
process.generator.PGunParameters.MinEta = -3.0
process.generator.PGunParameters.MaxEta = 3.0
process.generator.PGunParameters.MinPt = 50.0
process.generator.PGunParameters.MinPt = 51.0



# process.load("FastSimulation.Configuration.SimpleJet_cfi")

# Input source
#process.source = cms.Source("EmptySource")


"""
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#'/store/relval/CMSSW_3_1_0_pre10/RelValSingleTauPt50Pythia/GEN-SIM-RECO/IDEAL_31X_v1/0008/4C7D1339-5857-DE11-A513-0019B9F70607.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValSingleTauPt50Pythia/GEN-SIM-RECO/IDEAL_31X_v1/0008/3E1EE9AA-0458-DE11-BD9D-001D09F2960F.root'
  '/store/relval/CMSSW_3_1_0_pre10/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0009/0EC0E0FE-0558-DE11-986D-001D09F29146.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0008/969D21C1-E857-DE11-A00D-001D09F23D04.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0008/840866A4-EA57-DE11-8A75-001D09F28EC1.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0008/70EDDE9C-E957-DE11-A16D-000423D98868.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0001/CCE6F244-1458-DE11-A5E5-001A92971B5E.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0001/C4E56D8D-075A-DE11-B5E7-001A92971B54.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0001/16E3E391-7059-DE11-9887-001A928116F4.root'
  

        )

)
"""

#fastsim
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
process.GlobalTag.globaltag = "MC_31X_V1::All"


process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

process.famosSimHits.VertexGenerator.BetaStar = 0.00001
process.famosSimHits.VertexGenerator.SigmaZ = 0.00001

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# process.famosSimHits.MaterialEffects.PairProduction = False
# process.famosSimHits.MaterialEffects.Bremsstrahlung = False
# process.famosSimHits.MaterialEffects.EnergyLoss = False
# process.famosSimHits.MaterialEffects.MultipleScattering = False
# process.famosSimHits.MaterialEffects.NuclearInteraction = False

process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")
process.load("Validation.RecoParticleFlow.tauBenchmarkGeneric_cff")

process.p1 = cms.Path(
#    process.famosWithCaloTowersAndParticleFlow +
    process.generator +
    process.famosWithEverything +
    process.caloJetMetGen +
    process.particleFlowSimParticle
    )


process.load("FastSimulation.Configuration.EventContent_cff")
process.aod = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
#    fileName = cms.untracked.string('/storage/users/gennai/aodFastSim310pre10FromRelVal.root')
                                   fileName = cms.untracked.string('/storage/users/gennai/aodSingleTauPt50_310pre11.root')
)

process.load("FastSimulation.Configuration.EventContent_cff")
process.reco = cms.OutputModule("PoolOutputModule",
    process.RECOSIMEventContent,
    fileName = cms.untracked.string('/storage/users/gennai/recoFastSimSingleTauPt50_310pre11.root')
)

process.load("RecoParticleFlow.Configuration.Display_EventContent_cff")
process.display = cms.OutputModule("PoolOutputModule",
    process.DisplayEventContent,
    fileName = cms.untracked.string('display.root')
)

#process.outpath = cms.EndPath(process.aod + process.display)
process.outpath = cms.EndPath(process.reco)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(False),
    wantSummary = cms.untracked.bool(False),
    Rethrow = cms.untracked.vstring('Unknown', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ModuleFailure', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FatalRootError', 
        'NotFound')
)
process.MessageLogger.cerr.FwkReport.reportEvery = 100

