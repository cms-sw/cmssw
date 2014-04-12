# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
import os

from FWCore.ParameterSet.Modules import _Module
# Searches for self.lookFor module in cms.Path. When found, next and prev module is stored
class SeqVisitor(object):
    def __init__(self, lookFor):
	self.lookFor=lookFor
	self.nextInChain="NONE"
	self.prevInChain="NONE"
	self.prevInChainCandidate="NONE"
	self.catch=0   # 1 - we have found self.lookFor, at next visit write visitee
	self.found=0

    def prepareSearch(self): # this should be called on beggining of each iteration 
	self.found=0 
      
    def setLookFor(self, lookFor):
	self.lookFor = lookFor
      
    def giveNext(self):
	return self.nextInChain
    def givePrev(self):
	return self.prevInChain
      
    def enter(self,visitee):
	if isinstance(visitee, _Module):
	  if self.catch == 1:
	      self.catch=0
	      self.nextInChain=visitee
	      self.found=1
	  if visitee == self.lookFor:
	      self.catch=1
	      self.prevInChain=self.prevInChainCandidate
	      
	  self.prevInChainCandidate=visitee
	
    def leave(self,visitee):
	    pass

def customise(process):
  
  process._Process__name="EmbeddedRECO"
  process.TFileService = cms.Service("TFileService",  fileName = cms.string("histo_embedded.root")          )

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
                                 "keep *_generator_*_*",
                                 "keep *_tmfTracks_*_EmbeddedRECO",
                                 "keep *_offlinePrimaryVertices_*_EmbeddedRECO",
                                 "keep *_offlinePrimaryVerticesWithBS_*_EmbeddedRECO",
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
  RECOproc = "RECO"
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

  options.register ('transformationMode',
                    1, #default value
                    VarParsing.VarParsing.multiplicity.singleton,
                    VarParsing.VarParsing.varType.int,
                    "transformation mode. 0=mumu->mumu, 1=mumu->tautau")

  options.register ('minVisibleTransverseMomentum',
                    "", #default value
                    VarParsing.VarParsing.multiplicity.singleton,
                    VarParsing.VarParsing.varType.string,
                    "generator level cut on visible transverse momentum (typeN:pT,[...];[...])")

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

  options.register ('skimEnabled',
                    0, # default value, true
                    VarParsing.VarParsing.multiplicity.singleton,
                    VarParsing.VarParsing.varType.int,         
                    "should I apply Zmumu event selection cuts?")

#  options.register ('primaryProcess',
#                    'RECO', # default value
#                     VarParsing.VarParsing.multiplicity.singleton,
#                     VarParsing.VarParsing.varType.string,
#                     "original processName")


  setFromCL = False
  if not hasattr(process,"doNotParse"):
    import sys
    if hasattr(sys, "argv") == True:
      if not sys.argv[0].endswith('cmsDriver.py'):
        options.parseArguments()
        setFromCL = True
  else:
    print "CL parsing disabled!"
  if setFromCL:
    print "Setting mdtau to ", options.mdtau
    process.generator.ZTauTau.TauolaOptions.InputCards.mdtau = options.mdtau 
    process.newSource.ZTauTau.TauolaOptions.InputCards.mdtau = options.mdtau
    process.generator.ParticleGun.ExternalDecays.Tauola.InputCards.mdtau = options.mdtau 
    process.newSource.ParticleGun.ExternalDecays.Tauola.InputCards.mdtau = options.mdtau 

    print "Setting minVisibleTransverseMomentum to ", options.minVisibleTransverseMomentum
    process.newSource.ZTauTau.minVisibleTransverseMomentum = cms.untracked.string(options.minVisibleTransverseMomentum)
    process.generator.ZTauTau.minVisibleTransverseMomentum = cms.untracked.string(options.minVisibleTransverseMomentum)

    print "Setting transformationMode to ", options.transformationMode
    process.generator.ZTauTau.transformationMode = cms.untracked.int32(options.transformationMode)
    process.newSource.ZTauTau.transformationMode = cms.untracked.int32(options.transformationMode)

    print "options.overrideBeamSpot", options.overrideBeamSpot
    if options.overrideBeamSpot != 0:
      bs = cms.string("BeamSpotObjects_2009_LumiBased_SigmaZ_v26_offline") # 52x data PR gt
      # bs = cms.string("BeamSpotObjects_2009_LumiBased_SigmaZ_v21_offline") # 42x data PR gt
      # bs = cms.string("BeamSpotObjects_2009_LumiBased_SigmaZ_v18_offline") # 41x data PR gt
      # bs = cms.string("BeamSpotObjects_2009_LumiBased_v17_offline") # 38x data gt
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

# -*- coding: utf-8 -*-

  process.tmfTracks = cms.EDProducer("RecoTracksMixer",
      trackCol1 = cms.InputTag("removedInputMuons","tracks"),
      trackCol2 = cms.InputTag("generalTracks","","EmbeddedRECO")
  )  

  process.offlinePrimaryVerticesWithBS.TrackLabel = cms.InputTag("tmfTracks")
  process.offlinePrimaryVertices.TrackLabel = cms.InputTag("tmfTracks")

  if hasattr(process.muons, "TrackExtractorPSet"):
    process.muons.TrackExtractorPSet.inputTrackCollection = cms.InputTag("tmfTracks")
  elif hasattr(process, "muons1stStep") and hasattr(process.muons1stStep, "TrackExtractorPSet"):
    process.muons1stStep.TrackExtractorPSet.inputTrackCollection = cms.InputTag("tmfTracks")
  else:
    raise "Problem with muons"
  # it should be the best solution to take the original beam spot for the
  # reconstruction of the new primary vertex
  # use the  one produced earlier, do not produce your own
  for s in process.sequences:
     seq =  getattr(process,s)
     seq.remove(process.offlineBeamSpot) 

  try:
  	process.metreco.remove(process.BeamHaloId)
  except:
  	pass

  try:
	  outputModule = process.output
  except:
    pass
  try:
	  outputModule = getattr(process,str(getattr(process,list(process.endpaths)[-1])))
  except:
    pass

  for p in process.paths:
    pth = getattr(process,p)
    if "generalTracks" in pth.moduleNames():
      pth.replace(process.generalTracks, process.generalTracks*process.tmfTracks)

  #'''
  process.gsfElectronsORG = process.gsfElectrons.clone()
  #print dir(process)
  #for p in dir(process):

  for p in process.paths:
    pth = getattr(process,p)
    #if hasattr(pth,"gsfElectrons"):
    if "gsfElectrons" in pth.moduleNames():
      pth.replace(process.gsfElectrons, process.gsfElectronsORG*process.gsfElectrons)
      #print p, dir(pth.moduleNames())

  # xxx
  process.gsfElectrons = cms.EDProducer("GSFElectronsMixer",
      col1 = cms.InputTag("gsfElectronsORG"),
      col2 = cms.InputTag("gsfElectrons","", RECOproc )
  )
  #'''

  process.particleFlowORG = process.particleFlow.clone()

  # Since CMSSW 4_4 the particleFlow reco works a bit differently. The step is
  # twofold, first particleFlowTmp is created and then the final particleFlow
  # collection. What we do in this case is that we merge the final ParticleFlow
  # collection. For the muon reconstruction, we also merge particleFlowTmp in
  # order to get PF-based isolation right.
  if hasattr(process, 'particleFlowTmp'):
    process.particleFlowTmpMixed = cms.EDProducer('PFCandidateMixer',
      col1 = cms.untracked.InputTag("removedInputMuons","pfCands"),
      col2 = cms.untracked.InputTag("particleFlowTmp", ""),
      trackCol = cms.untracked.InputTag("tmfTracks"),

      # Don't produce value maps:
      muons = cms.untracked.InputTag(""),
      gsfElectrons = cms.untracked.InputTag("")
    )
    process.muons.PFCandidates = cms.InputTag("particleFlowTmpMixed")

    for p in process.paths:
      if "particleFlow" in pth.moduleNames():
        pth.replace(process.particleFlow, process.particleFlowORG*process.particleFlow)
      if "muons" in pth.moduleNames():
        pth.replace(process.muons, process.particleFlowTmpMixed*process.muons)
  else:
    # CMSSW_4_2
    if hasattr(process,"famosParticleFlowSequence"):
      process.famosParticleFlowSequence.remove(process.pfPhotonTranslatorSequence)
      process.famosParticleFlowSequence.remove(process.pfElectronTranslatorSequence)
      process.famosParticleFlowSequence.remove(process.particleFlow)
      process.famosParticleFlowSequence.__iadd__(process.particleFlowORG)
      process.famosParticleFlowSequence.__iadd__(process.particleFlow)
      process.famosParticleFlowSequence.__iadd__(process.pfElectronTranslatorSequence)
      process.famosParticleFlowSequence.__iadd__(process.pfPhotonTranslatorSequence)
    elif hasattr(process,"particleFlowReco"):
      process.particleFlowReco.remove(process.pfPhotonTranslatorSequence)
      process.particleFlowReco.remove(process.pfElectronTranslatorSequence)
      process.particleFlowReco.remove(process.particleFlow)
      process.particleFlowReco.__iadd__(process.particleFlowORG)
      process.particleFlowReco.__iadd__(process.particleFlow)
      process.particleFlowReco.__iadd__(process.pfElectronTranslatorSequence)
      process.particleFlowReco.__iadd__(process.pfPhotonTranslatorSequence)
    else :
      raise "Cannot find particleFlow sequence"

    process.pfSelectedElectrons.src = cms.InputTag("particleFlowORG")
    process.pfSelectedPhotons.src   = cms.InputTag("particleFlowORG")

  process.particleFlow = cms.EDProducer('PFCandidateMixer',
          col1 = cms.untracked.InputTag("removedInputMuons","pfCands"),
          col2 = cms.untracked.InputTag("particleFlowORG", ""),
          trackCol = cms.untracked.InputTag("tmfTracks"),

          muons = cms.untracked.InputTag("muons"),
          gsfElectrons = cms.untracked.InputTag("gsfElectrons")
          # TODO: photons???
  )

  process.filterEmptyEv.src = cms.untracked.InputTag("generator","","EmbeddedRECO")

  from FWCore.ParameterSet.Types import InputTag
  for p in process.paths:
     i =  getattr(process,p)
     target = process.particleFlow
     
     seqVis = SeqVisitor(target)
     seqVis.prepareSearch()
     seqVis.setLookFor(target)
     i.visit(seqVis) 
     while ( seqVis.catch != 1 and seqVis.found == 1 ): 

       target = seqVis.giveNext()

       targetAttributes =  dir(target)
       for targetAttribute in targetAttributes:
         attr=getattr(target,targetAttribute) # get actual attribute, not just  the name
         if isinstance(attr, InputTag) and attr.getModuleLabel()=="particleFlow":
           if ( attr.getProductInstanceLabel()!=""  ):
             print "Changing: ", target, " ", targetAttribute, " ", attr, " to particleFlowORG"
             attr.setModuleLabel("particleFlowORG")

       #i.replace(target, source) 
       seqVis.prepareSearch()
       seqVis.setLookFor(target)
       i.visit(seqVis) 
            
     #if (seqVis.catch==1):
       #seqVis.catch=0
       #i.__iadd__(source)

  # CV: mix L1Extra collections
  l1ExtraCollections = [
      [ "L1EmParticle",     "Isolated"    ],
      [ "L1EmParticle",     "NonIsolated" ],
      [ "L1EtMissParticle", "MET"         ],
      [ "L1EtMissParticle", "MHT"         ],
      [ "L1JetParticle",    "Central"     ],
      [ "L1JetParticle",    "Forward"     ],
      [ "L1JetParticle",    "Tau"         ],
      [ "L1MuonParticle",   ""            ]
  ]
  l1extraParticleCollections = []
  for l1ExtraCollection in l1ExtraCollections:
      inputType = l1ExtraCollection[0]
      pluginType = None
      if inputType == "L1EmParticle":
          pluginType = "L1ExtraEmParticleMixerPlugin"
      elif inputType == "L1EtMissParticle":
          pluginType = "L1ExtraMEtMixerPlugin"
      elif inputType == "L1JetParticle":
          pluginType = "L1ExtraJetParticleMixerPlugin"
      elif inputType == "L1MuonParticle":
          pluginType = "L1ExtraMuonParticleMixerPlugin"
      else:
          raise ValueError("Invalid L1Extra type = %s !!" % inputType)
      instanceLabel = l1ExtraCollection[1]
      l1extraParticleCollections.append(cms.PSet(
          pluginType = cms.string(pluginType),
          instanceLabel = cms.string(instanceLabel)))
  process.l1extraParticlesORG = process.l1extraParticles.clone()
  process.l1extraParticles = cms.EDProducer('L1ExtraMixer',
      src1 = cms.InputTag('l1extraParticles::HLT'),
      src2 = cms.InputTag('l1extraParticlesORG'),
      collections = cms.VPSet(l1extraParticleCollections)
  )
  for p in process.paths:
      pth = getattr(process,p)
      if "l1extraParticles" in pth.moduleNames():
          pth.replace(process.l1extraParticles, process.l1extraParticlesORG*process.l1extraParticles)

  #if hasattr(process, "DQM_FEDIntegrity_v3"):
  #  process.schedule.remove(process.DQM_FEDIntegrity_v3)
  fedIntRemoved = False
  for i in range(1,30):
     attrName = "DQM_FEDIntegrity_v"+str(i) 
     if hasattr(process, attrName ):
       process.schedule.remove(process.DQM_FEDIntegrity_v11)
       del process.DTDataIntegrityTask 
       fedIntRemoved = True
       break
 
  if fedIntRemoved:
      print "Removed", attrName
  else:
      print "DQM_FEDIntegrity_vXX not found, expect problems"

  skimEnabled = False
  if setFromCL:
      if options.skimEnabled != 0:
          skimEnabled = True
  else:
      if hasattr(process,"doZmumuSkim"):
          skimEnabled = True
  if skimEnabled:
      print "Enabling Zmumu skim"
      skimEnabled = True

      cmssw_ver = os.environ["CMSSW_VERSION"]
      if cmssw_ver.find("CMSSW_4_2") != -1:
        print
        print "Using legacy version of Zmumu skim. Note, that muon isolation is disabled"
        print
        process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandalonSelectionLegacy_cff")
        process.RandomNumberGeneratorService.dummy = cms.PSet(
          initialSeed = cms.untracked.uint32(123456789),
          engineName = cms.untracked.string('HepJamesRandom')
        )

      else:
        process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandalonSelection_cff")

      #process.load("TauAnalysis/Skimming/goldenZmmSelectionVBTFrelPFIsolation_cfi")
      process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")

      # we are allready selecting events from generation step, so following way is ok
      for path in process.paths:
          getattr(process,path)._seq = process.goldenZmumuSelectionSequence * getattr(process,path)._seq

      #process.options = cms.untracked.PSet(
      #  wantSummary = cms.untracked.bool(True)
      #)
  else:
      print "Zmumu skim not enabled"
      # CV: add path for keeping track of skimming efficiency
      process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandalonSelection_cff")
      process.skimEffFlag = cms.EDProducer("DummyBoolEventSelFlagProducer")
      process.skimEffPath = cms.Path(process.goldenZmumuSelectionSequence * process.skimEffFlag)
      
      process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")

  print "# ######################################################################################"
  print "  Following parameters can be added before customize function "
  print "  call in order to controll process  customization: "
  print "     process.doNotParse =  cms.PSet() # disables CL parsing for crab compat"
  print "     process.doZmumuSkim = cms.PSet() # adds Zmumu skimming before embedding is run"
  print "# ######################################################################################"

  return(process)
