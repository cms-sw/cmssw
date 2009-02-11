import FWCore.ParameterSet.Config as cms
def customise(process):

# geometry addition to avoid problems with ECAL chain
    
    process.load('Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi')
    process.load('Geometry.EcalMapping.EcalMapping_cfi')
    process.load('Geometry.EcalMapping.EcalMappingRecord_cfi')

# extend the particle gun acceptance

    process.source.AddAntiParticle = cms.untracked.bool(False)

# no magnetic field

    process.g4SimHits.UseMagneticField = cms.bool(False)
    process.UniformMagneticFieldESProducer = cms.ESProducer("UniformMagneticFieldESProducer",
                                                            ZFieldInTesla = cms.double(0.0)
                                                                )

    process.prefer("UniformMagneticFieldESProducer") 

# modify the content

    process.output.outputCommands.append("keep *_simHcalUnsuppressedDigis_*_*")

# user schedule: use only calorimeters digitization and local reconstruction

    del process.schedule[:]

    process.schedule.append(process.generation_step)
    process.schedule.append(process.simulation_step)

    process.ecalWeightUncalibRecHit.EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis")
    process.ecalWeightUncalibRecHit.EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis")
    process.ecalPreshowerRecHit.ESdigiCollection = cms.InputTag("simEcalPreshowerDigis")

    process.hbhereco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")
    process.horeco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")
    process.hfreco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")

    process.local_digireco = cms.Path(process.mix * process.calDigi * process.calolocalreco * process.caloTowersRec)

    process.schedule.append(process.local_digireco)

    process.load("Validation/Configuration/hcalSimValid_cff")
    process.local_validation = cms.Path(process.hcalSimValid)
    process.schedule.append(process.local_validation)

    process.schedule.append(process.endjob_step)
    process.schedule.append(process.out_step)
        
    return(process)
