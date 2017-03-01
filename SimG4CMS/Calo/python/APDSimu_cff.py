import FWCore.ParameterSet.Config as cms

def customise(process):

# add ECAL and HCAL specific Geant4 hits objects

    process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
        instanceLabel = cms.untracked.string('EcalValidInfo'),
        type = cms.string('EcalSimHitsValidProducer'),
        verbose = cms.untracked.bool(False)
    ))

# use directly the generator output, no Hector

    process.g4SimHits.Generator.HepMCProductLabel = cms.string('generatorSmeared')

# user schedule: use only calorimeters digitization and local reconstruction

    process.g4SimHits.ECalSD.StoreSecondary = True
    process.g4SimHits.CaloTrkProcessing.PutHistory = True
    process.simEcalUnsuppressedDigis.apdAddToBarrel = True

    return(process)
