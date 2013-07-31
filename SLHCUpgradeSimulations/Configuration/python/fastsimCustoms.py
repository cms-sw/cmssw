
def customiseDefault(process):
    process.pfTrack.TrajInEvents = cms.bool(True)
    process.csc2DRecHits.readBadChannels = cms.bool(False)
    return process

def customisePhase2(process):
    process=customiseDefault(process)
    process.famosSimHits.MaterialEffects.PairProduction = cms.bool(False)
    process.famosSimHits.MaterialEffects.Bremsstrahlung = cms.bool(False)
    process.famosSimHits.MaterialEffects.MuonBremsstrahlung = cms.bool(False)
    process.famosSimHits.MaterialEffects.EnergyLoss = cms.bool(False)
    process.famosSimHits.MaterialEffects.MultipleScattering = cms.bool(False)
    # keep NI so to allow thickness to be properly treated in the interaction geometry
    process.famosSimHits.MaterialEffects.NuclearInteraction = cms.bool(True)
    process.KFFittingSmootherWithOutlierRejection.EstimateCut = cms.double(50.0)
    return process
