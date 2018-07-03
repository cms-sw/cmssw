import FWCore.ParameterSet.Config as cms

def customise(process):

  #Adding SimpleMemoryCheck service:
  process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                        ignoreTotal=cms.untracked.int32(1),
                                        oncePerEventMode=cms.untracked.bool(True))
  #Adding Timing service:
  process.Timing=cms.Service("Timing")

  #Add these 3 lines to put back the summary for timing information at the end of the logfile
  #(needed for TimeReport report)
  if hasattr(process,'options'):
    process.options.wantSummary = cms.untracked.bool(True)
  else:
    process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
  )

  # a fragment allowing to disable various options
  if hasattr(process,'g4SimHits'):
    #  eta cuts
    process.g4SimHits.Generator.MinEtaCut = cms.double(-5.5)
    process.g4SimHits.Generator.MaxEtaCut = cms.double(5.5)
    #  Geant4 Physics List
    process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/FTFP_BERT')
    #  Russian roulette 
    process.g4SimHits.StackingAction.RusRoGammaEnergyLimit = cms.double(0.0)
    process.g4SimHits.StackingAction.RusRoNeutronEnergyLimit = cms.double(0.0)
    #  HF shower library
    process.g4SimHits.HCalSD.UseShowerLibrary = cms.bool(False)
    process.g4SimHits.HFShower.UseShowerLibrary = cms.bool(False)
    #  tracking cuts
    process.common_maximum_time.DeadRegions = cms.vstring()
    process.common_maximum_time.CriticalDensity = cms.double(0)
    process.common_maximum_time.CriticalEnergyForVacuum = cms.double(0)
    process.g4SimHits.StackingAction.TrackNeutrino = cms.bool(True)
    process.g4SimHits.StackingAction.KillGamma     = cms.bool(False)
    process.g4SimHits.StackingAction.CriticalEnergyForVacuum = cms.double(0)
    process.g4SimHits.StackingAction.DeadRegions = cms.vstring()
    process.g4SimHits.SteppingAction.CriticalDensity = cms.double(0)
    process.g4SimHits.SteppingAction.CriticalEnergyForVacuum = cms.double(0)
    process.g4SimHits.SteppingAction.DeadRegions = cms.vstring()
    #  time cuts
    TimeCut = cms.double(10000.0)
    process.common_maximum_time.MaxTrackTime = TimeCut
    process.g4SimHits.StackingAction.MaxTrackTime = TimeCut
    process.g4SimHits.SteppingAction.MaxTrackTime = TimeCut
    process.g4SimHits.CaloSD.TmaxHit = TimeCut
    process.g4SimHits.CaloSD.TmaxHits = cms.vdouble(10000,10000,10000,10000,10000)
    #  cuts per region
    process.g4SimHits.Physics.CutsPerRegion = cms.bool(False)
    process.g4SimHits.Physics.DefaultCutValue = cms.double(0.07)
    return(process)
