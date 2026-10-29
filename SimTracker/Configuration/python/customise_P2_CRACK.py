####################################################################
#
# To create Monte Carlo of cosmic rays traversing the
# PhaseII Tracker Cosmic Track, do:
#
# cmsDriver.py UndergroundCosmicSPLooseMu_cfi -s GEN,SIM -n 10000 --conditions auto:phase2_realistic_0T --beamspot DBrealisticHLLHC --datatier GEN-SIM --eventcontent FEVTDEBUG --geometry ExtendedRun4D500 --era phase2_tracker --fileout file:step1.root --nThreads 16 --python step1.py --customise SimTracker/Configuration/customise_P2_CRACK.step1
#
# cmsDriver.py -s DIGI:pdigi_valid,L1TrackTrigger --conditions auto:phase2_realistic_0T --datatier GEN-SIM-DIGI --eventcontent FEVTDEBUG --geometry ExtendedRun4D500 --era phase2_tracker --magField 0T -n -1 --filein file:step1.root --fileout file:step2.root --nThreads 16  --python step2.py --customise SimTracker/Configuration/customise_P2_CRACK.step2
#
# See also Geometry/TrackerCommonData/doc/README_CRACK_P2.md
#
# (This was last validated in CMSSW_20_1_0_pre1)
#
####################################################################

#===== This does GEN-SIM step =====

def step1(process):

  # Modify generator so no material above CRACK
  process.generator.ElossScaleFactor = 0.0
  # Specify target volume. Cylinder of this radius & half-length must
  # contain CRACK (which has r < 60cm, 0 < z < 120cm).
  # This requires cosmics to cross CMS Tracker volume (r < 120cm, |z| < 280cm)
  # and generates them on Tracker surface, so their TOF is OK.
  process.generator.TrackerOnly = True
  # Max angle to vertical of cosmics (given by CRACK trigger acceptance)
  process.generator.MaxTheta = 45.0

  # Use cosmic filter to reduce empty events.
  process.cosmicInPixelLoose.radius = 600.0
  process.cosmicInPixelLoose.minZ = -10.0
  process.cosmicInPixelLoose.maxZ = 130.0

  # Use ideal alignment, since GlobalTag knows nothing about alignment of CRACK
  process.trackerGeometry.applyAlignment = False

  # Limit simulation to Tracker
  process.g4SimHits.OnlySDs = ['TkAccumulatingSensitiveDetector']

  return process

#===== This does DIGI + cluster, TTCluster, TTStub steps ======

def step2(process):

  # Disable L1 tracking algo, as doesn't make sense for CRACK.
  process.L1TrackTrigger.remove(process.L1TPromptExtendedHybridTracksWithAssociators)

  # Disable DTC emulation as it needs CRack cabling map.
  process.L1TrackTrigger.remove(process.ProducerDTC)

  # But do run offline cluster finding
  process.load('RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi')
  process.L1TrackTrigger += process.siPhase2Clusters


  # Use ideal alignment, since GlobalTag knows nothing about alignment of CRACK
  process.trackerGeometry.applyAlignment = False

  # Don't get Lorentz angle from GlobalTag, since it knows nothing about CRACK.

  process.mix.digitizers.pixel.SSDigitizerAlgorithm.LorentzAngle_DB = False
  process.mix.digitizers.pixel.PSSDigitizerAlgorithm.LorentzAngle_DB = False
  process.mix.digitizers.pixel.PSPDigitizerAlgorithm.LorentzAngle_DB = False

  # Relax stub bend cuts in FE electronics as much as possible, as cosmics.
  # (N.B. OPTION NOT YET IN CENTRAL CMSSW)
  if hasattr(process.TTStubAlgorithm_official_Phase2TrackerDigi_, "cosmics"):
    process.TTStubAlgorithm_official_Phase2TrackerDigi_.cosmics = True
  else:
    print("WARNING: TTStub cosmic cfg option not available in this CMSSW release.")

  #--- Remove non-Tracker digitisation

  print("REMOVING NON-TRACKER DIGISATION")

  # Drop digitisation modules for calo etc.
  process.pdigiTask_nogen = process.pdigiTask_nogen.copyAndExclude(
      [process.doAllDigiTask]
  )
  
  # Remove calo etc. digitsation from mixing module.
  cont = process.mix.digitizers
  keep = {"pixel", "puVtx", "mergedtruth"}
  for k in list(cont.parameters_().keys()):
      if k not in keep:
          print("Removed mix.digitizers key ", k)
          delattr(cont, k)
      else:
          print("Kept mix.digitizers key ", k)   

  # Drop any EDAlias that points to non-Tracker digis from 'mix'
  def _drop_aliases_pointing_to_mix_nonTracker(process):
      keep = {"simSiPixelDigis"}
      for name, alias in list(process.aliases_().items()):
          print ("name is ", name , "alias is ", alias )
          if name not in keep :
            delattr(process, name)
            print("[tracker-only] Removed EDAlias:", name)
      return process

  process = _drop_aliases_pointing_to_mix_nonTracker(process)         

  return process
