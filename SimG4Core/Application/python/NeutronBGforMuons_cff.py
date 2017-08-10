import FWCore.ParameterSet.Config as cms

def neutronBG(process):

  # common fragment allowing to simulate neutron background in muon system

  if hasattr(process,'g4SimHits'):
  # time window 10 second
    TimeCut = cms.double(10000000000.0)
    process.common_maximum_time.MaxTrackTime = TimeCut
    process.common_maximum_time.DeadRegions = cms.vstring()
    # Physics List XS
    process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/FTFP_BERT_XS_EML')
    process.g4SimHits.Physics.CutsOnProton  = cms.untracked.bool(False)
    process.g4SimHits.Physics.FlagFluo    = cms.bool(True)
    process.g4SimHits.Physics.ThermalNeutrons = cms.untracked.bool(False)
    # Eta cut
    process.g4SimHits.Generator.MinEtaCut = cms.double(-7.0)
    process.g4SimHits.Generator.MaxEtaCut = cms.double(7.0)
    # stacking action
    process.g4SimHits.StackingAction.MaxTrackTime = TimeCut
    process.g4SimHits.StackingAction.DeadRegions = cms.vstring()
    process.g4SimHits.StackingAction.GammaThreshold = cms.double(0.0)
    # stepping action
    process.g4SimHits.SteppingAction.MaxTrackTime = TimeCut
    process.g4SimHits.SteppingAction.DeadRegions = cms.vstring()
    # Russian roulette disabled
    process.g4SimHits.StackingAction.RusRoGammaEnergyLimit = cms.double(0.0)
    process.g4SimHits.StackingAction.RusRoNeutronEnergyLimit = cms.double(0.0)
    # Calorimeter hits
    process.g4SimHits.CaloSD.TmaxHit = TimeCut
    process.g4SimHits.CaloSD.TmaxHits = cms.vdouble(10000000000,10000000000,10000000000,10000000000,10000000000)
    # full simulation of HF 
    process.g4SimHits.HCalSD.UseShowerLibrary = cms.bool(False)
    process.g4SimHits.HFShower.UseShowerLibrary = cms.bool(False)

    return(process)
