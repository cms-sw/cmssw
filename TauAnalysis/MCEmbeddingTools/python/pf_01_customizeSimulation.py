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
  
  
  process._Process__name="SELECTandSIM"


  process.TFileService = cms.Service("TFileService",  fileName = cms.string("histo_reconstruction.root")          )

  process.tmfTracks = cms.EDProducer("RecoTracksMixer",
      trackCol1 = cms.InputTag("dimuonsGlobal"),
      trackCol2 = cms.InputTag("generalTracks","","SELECTandSIM")
  )  

  process.offlinePrimaryVerticesWithBS.TrackLabel = cms.InputTag("tmfTracks")
  process.offlinePrimaryVertices.TrackLabel = cms.InputTag("tmfTracks")
  process.muons.TrackExtractorPSet.inputTrackCollection = cms.InputTag("tmfTracks")
  # I don't know why I added the following two lines, so I disable them for the moment
  #process.offlinePrimaryVertices.beamSpotLabel = cms.InputTag("offlineBeamSpot","","RECO")
  #process.offlinePrimaryVerticesWithBS.beamSpotLabel = cms.InputTag("offlineBeamSpot","","RECO")
  
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

  print "Changing eventcontent to AODSIM + misc "
  outputModule.outputCommands = process.AODSIMEventContent.outputCommands
  keepMC = cms.untracked.vstring("keep *_*_zMusExtracted_*",
                                 "keep *_dimuonsGlobal_*_*",
                                 'keep *_generator_*_*',
                                 "keep *_tmfTracks_*_SELECTandSIM",
                                 "keep *_offlinePrimaryVertices_*_SELECTandSIM",
                                 "keep *_offlinePrimaryVerticesWithBS_*_SELECTandSIM",
  )
  outputModule.outputCommands.extend(keepMC)

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

  process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

  import FWCore.ParameterSet.VarParsing as VarParsing
  options = VarParsing.VarParsing ('analysis')

  options.register ('overrideBeamSpot',
                    0, # default value, false
                    VarParsing.VarParsing.multiplicity.singleton,
                    VarParsing.VarParsing.varType.int,
                    "should I override beamspot in globaltag?")


  if options.overrideBeamSpot !=  0:
    bs = cms.string("BeamSpotObjects_2009_LumiBased_v16_offline") # 38x data gt
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

  print "#############################################################"
  print " Warning! PFCandidates 'electron' collection is not mixed, "
  print "  and probably shouldnt be used. "
  print "#############################################################"
  return(process)
