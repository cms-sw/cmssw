


def customise(process):
  process._Process__name="SELECTandSIM"

  process.TFileService = cms.Service("TFileService",  fileName = cms.string("histo.root")          )

  process.tmfTracks = cms.EDProducer("RecoTracksMixer",
      trackCol1 = cms.InputTag("dimuonsGlobal"),
      trackCol2 = cms.InputTag("generalTracks","","SELECTandSIM")
  )  

  process.offlinePrimaryVerticesWithBS.TrackLabel = cms.InputTag("tmfTracks")
  process.offlinePrimaryVertices.TrackLabel = cms.InputTag("tmfTracks")
  

  from FWCore.ParameterSet.Modules import _Module
  class SeqVisitor(object):
      def __init__(self, lookFor):
         self.lookFor=lookFor
         self.nextInChain="NONE"
         self.catch=0
         self.found=0

      def prepareSearch(self): # this should be called on beggining of each iteration 
         self.found=0 
       
      def setLookFor(self, lookFor):
         self.lookFor = lookFor
       
      def giveNext(self):
         return self.nextInChain
       
      def enter(self,visitee):
         if isinstance(visitee, _Module):
            if self.catch == 1:
               self.catch=0
               self.nextInChain=visitee
               self.found=1
            if visitee == self.lookFor:
               self.catch=1
         
      def leave(self,visitee):
             pass
                                            

  for p in process.paths:
     i =  getattr(process,p)
     target = process.generalTracks
     source = process.tmfTracks 
##
     seqVis = SeqVisitor(source)
     seqVis.prepareSearch()
     seqVis.setLookFor(target)
     i.visit(seqVis) # finds next module in path after self.lookFor
   
  
     while ( seqVis.catch != 1 and seqVis.found == 1 ): # the module we are looking for is allready at the end of path
       target = seqVis.giveNext()
       #     print "Replaceing " + target.label() + " with " + source.label()
       i.replace(target, source) # replace target with source 
       seqVis.prepareSearch()
       seqVis.setLookFor(source)
       i.visit(seqVis) # finds next module in path after "source"
       source = target # prepare to replace module we have just found  with the module we have overwritten
     
     if (seqVis.catch==1):
       seqVis.catch=0
       #     print "Adding " + source.label() + " to path"
       i.__iadd__(source)
       
  return(process)

