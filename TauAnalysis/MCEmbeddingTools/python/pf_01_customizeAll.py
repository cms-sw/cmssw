# -*- coding: utf-8 -*-


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
                                 "keep *_dimuonsGlobal_*_*",
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

  options.register ('primaryProcess',
                    'RECO', # default value
                     VarParsing.VarParsing.multiplicity.singleton,
                     VarParsing.VarParsing.varType.string,
                     "original processName")



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
# -*- coding: utf-8 -*-


  process.tmfTracks = cms.EDProducer("RecoTracksMixer",
      trackCol1 = cms.InputTag("dimuonsGlobal"),
      trackCol2 = cms.InputTag("generalTracks","","SELECTandSIM")
  )  

  process.offlinePrimaryVerticesWithBS.TrackLabel = cms.InputTag("tmfTracks")
  process.offlinePrimaryVertices.TrackLabel = cms.InputTag("tmfTracks")
  process.muons.TrackExtractorPSet.inputTrackCollection = cms.InputTag("tmfTracks")
  # it should be the best solution to take the original beam spot for the
  # reconstruction of the new primary vertex
  #process.offlinePrimaryVertices.beamSpotLabel = cms.InputTag("offlineBeamSpot","","RECO")
  #process.offlinePrimaryVerticesWithBS.beamSpotLabel = cms.InputTag("offlineBeamSpot","","RECO")
  process.offlinePrimaryVertices.beamSpotLabel = cms.InputTag("offlineBeamSpot","","REDIGI311X")
  process.offlinePrimaryVerticesWithBS.beamSpotLabel = cms.InputTag("offlineBeamSpot","","REDIGI311X")
  
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


  if  hasattr(process,"iterativeTracking" ) :
    process.iterativeTracking.__iadd__(process.tmfTracks)
  elif hasattr(process,"trackCollectionMerging" ) :
    process.trackCollectionMerging.__iadd__(process.tmfTracks)
  else :
    raise "Cannot find tracking sequence"

  process.particleFlowORG = process.particleFlow.clone()
  if hasattr(process,"famosParticleFlowSequence"):
    process.famosParticleFlowSequence.remove(process.pfElectronTranslatorSequence)
    process.famosParticleFlowSequence.remove(process.particleFlow)
    process.famosParticleFlowSequence.__iadd__(process.particleFlowORG)
    process.famosParticleFlowSequence.__iadd__(process.particleFlow)
    process.famosParticleFlowSequence.__iadd__(process.pfElectronTranslatorSequence)
  elif hasattr(process,"particleFlowReco"):
    process.particleFlowReco.remove(process.pfElectronTranslatorSequence)
    process.particleFlowReco.remove(process.particleFlow)
    process.particleFlowReco.__iadd__(process.particleFlowORG)
    process.particleFlowReco.__iadd__(process.particleFlow)
    process.particleFlowReco.__iadd__(process.pfElectronTranslatorSequence)
  else :
    raise "Cannot find tracking sequence"

  process.particleFlow =  cms.EDProducer('PFCandidateMixer',
          col1 = cms.untracked.InputTag("dimuonsGlobal","forMixing"),
          col2 = cms.untracked.InputTag("particleFlowORG", "")
  )

 

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
             print "Changing: ", target, " ", targetAttribute, " ", attr, " to particleFlowORG", 
             attr.setModuleLabel("particleFlowORG")


       #i.replace(target, source) 
       seqVis.prepareSearch()
       seqVis.setLookFor(target)
       i.visit(seqVis) 
            
     #if (seqVis.catch==1):
       #seqVis.catch=0
       #i.__iadd__(source)


  print "#############################################################"
  print " Warning! PFCandidates 'electron' collection is not mixed, "
  print "  and probably shouldnt be used. "
  print "#############################################################"
  return(process)
