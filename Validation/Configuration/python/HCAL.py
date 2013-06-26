import FWCore.ParameterSet.Config as cms
def customise(process):

# extend the particle gun acceptance

    process.generator.AddAntiParticle = cms.bool(False)

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

    delattr(process,"hbhereco")
    process.hbhereco = process.hbheprereco.clone()
    process.hcalLocalRecoSequence.replace(process.hbheprereco,process.hbhereco)
    process.hbhereco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")
    process.horeco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")
    process.hfreco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")

    process.local_digireco = cms.Path(process.mix * process.hcalDigiSequence * process.hbhereco * process.hfreco * process.horeco )

    process.schedule.append(process.local_digireco)

    process.load("Validation/Configuration/hcalSimValid_cff")

    process.AllRecHitsValidation.ecalselector = cms.untracked.string('no')
    
    process.local_validation = cms.Path(process.hcalSimHitStudy+process.hcalDigisValidationSequence+process.hcalRecHitsValidationSequence)
    process.schedule.append(process.local_validation)

    process.schedule.append(process.endjob_step)
    process.schedule.append(process.out_step)
        
    return(process)
