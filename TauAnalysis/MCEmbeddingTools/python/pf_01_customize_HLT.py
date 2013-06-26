# -*- coding: utf-8 -*-

from FWCore.ParameterSet.Modules import _Module


def customise(process):
 
   
  
  process._Process__name="HLT2"
  process.TFileService = cms.Service("TFileService",  fileName = cms.string("histo_simulation.root")          )

  # Add the production filter sequence to all paths so that the empty event
  # filter is actually applied and events with no HepMC product are skipped.
  # Also add process.pgen for the generation_step which is not inserted by
  # cmsDriver for some reason.
  for path in process.paths: 
    if path != 'generation_step':
      getattr(process,path)._seq = process.ProductionFilterSequence*getattr(process,path)._seq
    else:
      getattr(process,path)._seq = getattr(process,path)._seq*cms.Sequence(process.pgen)

  try:
	  outputModule = process.output
  except:
    pass
  try:
	  outputModule = getattr(process,str(getattr(process,list(process.endpaths)[-1])))
  except:
    pass

  print "Changing eventcontent to RAW+AODSIM + misc. "
  outputModule.outputCommands = cms.untracked.vstring("drop *")
  outputModule.outputCommands.extend(process.RAWEventContent.outputCommands )
  outputModule.outputCommands.extend(process.AODSIMEventContent.outputCommands )

  keepMC = cms.untracked.vstring("keep *_*_zMusExtracted_*",
                                 "keep *_*_zmmCands_*",
                                 "keep *_removedInputMuons_*_*",
                                 "keep *_generator_*_*",
                                 "keep *_PhotonIDProd_*_*",
                                 "keep *_photons_*_*",
                                 "keep *_photonCore_*_*",
                                 "keep *_genParticles_*_*",
                                 "keep *_particleFlow_*_*",
  )
  outputModule.outputCommands.extend(keepMC)

  # getRid of second "drop *"
  index = 0
  for item in outputModule.outputCommands[:]:
    if item == "drop *" and index != 0:
      #print index," ",outputModule.outputCommands[index]
      del outputModule.outputCommands[index]
      index -= 1
    index += 1  


  hltProcessName = "HLT"	#"REDIGI38X"
  # the following block can be used for more efficient processing by replacing the HLT variable below automatically
  try:
    hltProcessName = __HLT__
  except:
    pass
	
  try:
    process.dimuonsHLTFilter.TriggerResultsTag.processName = hltProcessName
    process.goodZToMuMuAtLeast1HLT.TrigTag.processName = hltProcessName
    process.goodZToMuMuAtLeast1HLT.triggerEvent.processName = hltProcessName
    process.hltTrigReport,HLTriggerResults.processName = hltProcessName
  except:
    pass

  process.VtxSmeared = cms.EDProducer("FlatEvtVtxGenerator", 
    MaxZ = cms.double(0.0),
    MaxX = cms.double(0.0),
    MaxY = cms.double(0.0),
    MinX = cms.double(0.0),
    MinY = cms.double(0.0),
    MinZ = cms.double(0.0),
    TimeOffset = cms.double(0.0),
    src = cms.InputTag("generator")
  )

  import FWCore.ParameterSet.VarParsing as VarParsing
  options = VarParsing.VarParsing ('analysis')
  options.register ('mdtau',
                    0, # default value
                    VarParsing.VarParsing.multiplicity.singleton,
                    VarParsing.VarParsing.varType.int,         
                    "mdtau value for tauola")

  options.register ('useJson',
                    0, # default value, false
                    VarParsing.VarParsing.multiplicity.singleton,
                    VarParsing.VarParsing.varType.int,         
                    "should I enable json usage?")

  options.register ('overrideBeamSpot',
                    0, # default value, false
                    VarParsing.VarParsing.multiplicity.singleton,
                    VarParsing.VarParsing.varType.int,         
                    "should I override beamspot in globaltag?")

  # Set this to REDIGI311X for Spring11 MC:
  # (not actually used in this step but we define it so that both config files
  # can be called with the same arguments):
  options.register ('primaryProcess',
                    'RECO', # default value
                    VarParsing.VarParsing.multiplicity.singleton,
                    VarParsing.VarParsing.varType.string,
                    "original processName")

  # Workaround so that edmConfigHash does not fail with this config file.
  # cf. https://hypernews.cern.ch/HyperNews/CMS/get/crabFeedback/3852/1/1/1/1/1.html
  import sys
  if hasattr(sys, "argv") == True:
    options.parseArguments()

  print "Setting mdtau to ", options.mdtau
  process.generator.ZTauTau.TauolaOptions.InputCards.mdtau = options.mdtau 
  process.newSource.ZTauTau.TauolaOptions.InputCards.mdtau = options.mdtau
  process.generator.ParticleGun.ExternalDecays.Tauola.InputCards.mdtau = options.mdtau 
  process.newSource.ParticleGun.ExternalDecays.Tauola.InputCards.mdtau = options.mdtau 

  print "options.overrideBeamSpot", options.overrideBeamSpot
  if options.overrideBeamSpot != 0:
    bs = cms.string("BeamSpotObjects_2009_LumiBased_SigmaZ_v18_offline") # 41x data PR gt
    #bs = cms.string("BeamSpotObjects_2009_LumiBased_v17_offline") # 38x data gt
    #bs = cms.string("BeamSpotObjects_2009_v14_offline") # 36x data gt
    #  tag = cms.string("Early10TeVCollision_3p8cm_31X_v1_mc_START"), # 35 default
    #  tag = cms.string("Realistic900GeVCollisions_10cm_STARTUP_v1_mc"), # 36 default
    process.GlobalTag.toGet = cms.VPSet(
      cms.PSet(record = cms.string("BeamSpotObjectsRcd"),
           tag = bs,
           connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_BEAMSPOT")
      )
    )
    print "BeamSpot in globaltag set to ", bs 
  else:
    print "BeamSpot in globaltag not changed"

  if options.useJson !=  0:
    print "Enabling json usage"
    import PhysicsTools.PythonAnalysis.LumiList as LumiList
    import FWCore.ParameterSet.Types as CfgTypes
    myLumis = LumiList.LumiList(filename = 'my.json').getCMSSWString().split(',')
    process.source.lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
    process.source.lumisToProcess.extend(myLumis)

  try:
    process.newSource.ZTauTau.minVisibleTransverseMomentum = cms.untracked.double(__MINVISPT__)
    process.generator.ZTauTau.minVisibleTransverseMomentum = cms.untracked.double(__MINVISPT__)
  except:
    pass
  try:
    process.generator.ZTauTau.TauolaOptions.InputCards.mdtau = __MDTAU__
    process.generator.ParticleGun.ExternalDecays.Tauola.InputCards.mdtau = __MDTAU__
    process.newSource.ZTauTau.TauolaOptions.InputCards.mdtau = __MDTAU__
    process.newSource.ParticleGun.ExternalDecays.Tauola.InputCards.mdtau = __MDTAU__
  except:
    pass

  try:
    process.generator.ZTauTau.transformationMode = cms.untracked.int32(__TRANSFORMATIONMODE__)
    process.generator.ZTauTau.transformationMode = cms.untracked.int32(__TRANSFORMATIONMODE__)
  except:
    pass

  return(process)
