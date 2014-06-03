import FWCore.ParameterSet.Config as cms

def customise(process):

    # fragment allowing to simulate neutron background in muon system

  if hasattr(process,'g4SimHits'):
    # time window 10 millisecond
    process.common_maximum_time.MaxTrackTime = cms.double(100000000.0)
    process.common_maximum_time.DeadRegions = cms.vstring()
    # Physics List XS
    process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/FTFP_BERT_HP_EML')
    process.g4SimHits.Physics.CutsOnProton  = cms.untracked.bool(False)
    process.g4SimHits.Physics.FlagMuNucl  = cms.bool(True)
    process.g4SimHits.Physics.FlagFluo    = cms.bool(True)
    # Eta cut
    process.g4SimHits.Generator.MinEtaCut = cms.double(-7.0)
    process.g4SimHits.Generator.MaxEtaCut = cms.double(7.0)
    # stacking action
    process.g4SimHits.StackingAction.MaxTrackTime = cms.double(100000000.0)
    process.g4SimHits.StackingAction.DeadRegions = cms.vstring()
    # stepping action
    process.g4SimHits.SteppingAction.MaxTrackTime = cms.double(100000000.0)
    process.g4SimHits.SteppingAction.DeadRegions = cms.vstring()
    # Russian roulette disabled
    process.g4SimHits.StackingAction.RusRoGammaEnergyLimit = cms.double(0.0)
    process.g4SimHits.StackingAction.RusRoNeutronEnergyLimit = cms.double(0.0)
    # full simulation of HF 
    process.g4SimHits.HFShower.UseHFGflash = cms.bool(False)

    return(process)
