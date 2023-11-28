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

# add ECAL and HCAL specific Geant4 hits objects

    process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
        instanceLabel = cms.untracked.string('EcalValidInfo'),
        type = cms.string('EcalSimHitsValidProducer'),
        verbose = cms.untracked.bool(False)
    ))

    # use directly the generator output, no Hector

    process.g4SimHits.Generator.HepMCProductLabel = 'generatorSmeared'

    # modify the content

    #process.output.outputCommands.append("keep *_simHcalUnsuppressedDigis_*_*")
    next(iter(process.outputModules_().items()))[1].outputCommands.append("keep *_simHcalUnsuppressedDigis_*_*")
# user schedule: use only calorimeters digitization and local reconstruction

    del process.schedule[:] 

    process.schedule.append(process.generation_step)
    process.schedule.append(process.simulation_step)

    process.ecalMultiFitUncalibRecHit.cpu.EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis")
    process.ecalMultiFitUncalibRecHit.cpu.EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis")
    process.ecalPreshowerRecHit.ESdigiCollection = cms.InputTag("simEcalPreshowerDigis") 

    delattr(process,"hbhereco")
    process.hbhereco = process.hbheprereco.clone()
    process.hcalLocalRecoSequence.replace(process.hbheprereco,process.hbhereco)
    process.hbhereco.cpu.digiLabelQIE8 = cms.InputTag("simHcalUnsuppressedDigis")
    process.hbhereco.cpu.digiLabelQIE11 = cms.InputTag("simHcalUnsuppressedDigis","HBHEQIE11DigiCollection")
    process.horeco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")
    process.hfreco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")
    process.ecalRecHit.cpu.recoverEBIsolatedChannels = cms.bool(False)
    process.ecalRecHit.cpu.recoverEEIsolatedChannels = cms.bool(False)
    process.ecalRecHit.cpu.recoverEBFE = cms.bool(False)
    process.ecalRecHit.cpu.recoverEEFE = cms.bool(False)

#    process.local_digireco = cms.Path(process.mix * process.calDigi * process.ecalLocalRecoSequence * process.hbhereco * process.hfreco * process.horeco * (process.ecalClusters+process.caloTowersRec) * process.reducedEcalRecHitsSequence )

    process.reducedEcalRecHitsEB.interestingDetIdCollections = cms.VInputTag(
            # ecal
            cms.InputTag("interestingEcalDetIdEB"),
            cms.InputTag("interestingEcalDetIdEBU"),
            )
            
    process.reducedEcalRecHitsEE.interestingDetIdCollections = cms.VInputTag(
            # ecal
            cms.InputTag("interestingEcalDetIdEE"),
            )            

    process.local_digireco = cms.Path(process.mix * process.addPileupInfo * process.bunchSpacingProducer * process.calDigi * process.ecalPacker * process.esDigiToRaw * process.hcalRawData * process.rawDataCollector * process.ecalDigis * process.ecalPreshowerDigis * process.hcalDigis * process.calolocalreco * process.hbhereco * process.hfreco * process.horeco *(process.ecalClustersNoPFBox+process.caloTowersRec) * process.reducedEcalRecHitsSequenceEcalOnly )

    process.schedule.append(process.local_digireco)

    # add HcalNoiseRBXCollection product for Validation/CaloTowers Validation/HcalRecHits  
    process.load( "RecoMET.METProducers.hcalnoiseinfoproducer_cfi" )
    process.hcalnoise_path = cms.Path( process.hcalnoise )
    process.schedule.append( process.hcalnoise_path )

    process.load("Validation/Configuration/ecalSimValid_cff") 
    process.load("Validation/Configuration/hcalSimValid_cff") 
    process.local_validation = cms.Path(process.ecalSimValid+process.hcalSimValid)
    process.schedule.append(process.local_validation) 

    process.schedule.append(process.endjob_step)
    #process.schedule.append(process.out_step)

    return(process)
