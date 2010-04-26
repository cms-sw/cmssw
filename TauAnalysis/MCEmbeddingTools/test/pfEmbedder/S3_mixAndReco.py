import FWCore.ParameterSet.Config as cms

process = cms.Process("mixAndReco")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("RecoTauTag/Configuration/RecoPFTauTag_cff")
#process.load("RecoParticleFlow/Configuration/RecoParticleFlow_cff")
process.load("RecoMET/Configuration/RecoPFMET_cff")
process.load("RecoJets/Configuration/RecoPFJets_cff")

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')

#process.GlobalTag.globaltag = 'MC_3XY_V9A::All'
#process.GlobalTag.globaltag = 'STARTUP3X_V8A::All'
#process.GlobalTag.globaltag = 'STARTUP3X_V8A::All'
#process.GlobalTag.globaltag = 'STARTUP31X_V4::All' # 31x
#process.GlobalTag.globaltag = 'MC_31X_V5::All' # 31x
process.GlobalTag.globaltag = 'MC_3XY_V26::All' # 356


process.source = cms.Source("PoolSource",
        #skipBadFiles = cms.untracked.bool(True),
        skipEvents = cms.untracked.uint32(0),
        fileNames = cms.untracked.vstring(
#'file:/tmp/fruboes/Zmumu/step2a_RAW2DIGI_RECO.root'
               'file:/tmp/fruboes/Zmumu/s2_FASTSIM.root'
        )
#fileNames = cms.untracked.vstring(),
#inputCommands = cms.untracked.vstring(
#            "keep recoPFCandidates_dimuonsGlobal_*_*",
#            "keep recoPFCandidates_particleFlow_*_*" 
#        )
)

process.load("Configuration.EventContent.EventContent_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = 'DEBUG'
process.MessageLogger.destinations = cms.untracked.vstring('log2')
#process.MessageLogger.log = cms.untracked.PSet( threshold = cms.untracked.string("DEBUG") )
#process.MessageLogger.debugModules = cms.untracked.vstring("*")
    


process.OUTPUT = cms.OutputModule("PoolOutputModule",
#outputCommands = cms.untracked.vstring("drop *", 
#         "keep *_*_*_mixAndReco",
#         "keep *_*_*_EXAMPLE",
#         "keep *_offlinePrimaryVerticesWithBS_*_*",
#         "keep recoVertexs_offlinePrimaryVertices__HLT2",
#         "keep recoTracks_*_*_*",
#         "keep edmTriggerResults_TriggerResults__HLT2",
#         "keep *_generalTracks_*_HLT2",
#         "keep recoGsfElectrons_gsfElectrons__HLT2",
#         "keep recoCaloTaus_caloRecoTauProducer__HLT2",
#         "keep recoMuons_muons__HLT2",
#         "keep *_*_*_HLT2" 
#       ),
#SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p1')),
#     fileName = cms.untracked.string('file:/tmp/fruboes/Zmumu/st3_mixedPFCollection.root')
    fileName = cms.untracked.string('st3_mixedPFCollection.root')
)


process.particleFlow =  cms.EDProducer('PFCandidateMixer',
    col1 = cms.untracked.InputTag("dimuonsGlobal","forMixing"),
    col2 = cms.untracked.InputTag("particleFlow", "")
)

#process.pfMet.src = cms.InputTag("mixPF")
process.pfMet.src = cms.InputTag("particleFlow","","mixAndReco")

#####################################  
######### inputTag mod:
#####################################  
from FWCore.ParameterSet.Modules import EDProducer
class SeqVisitor(object):
    def enter(self,visitee):
       if isinstance(visitee,EDProducer):
            #print "Changing",visitee.label_(),"src to mixPF"
             #visitee.src = cms.InputTag("mixPF")
             visitee.src = cms.InputTag("particleFlow","","mixAndReco")
         
            #raise ValueError("Path cannot have an OutputModule, "+visitee.label_())
    def leave(self,visitee):
          pass
                                            
seqVis = SeqVisitor()
process.recoPFJets.visit(seqVis)
##############################  
##############################  
##############################  

process.p1 = cms.Path(process.particleFlow 
    * process.recoPFMET 
#* process.recoAllPFJets 
    * process.recoPFJets 
    * process.PFTau)

process.outpath = cms.EndPath(process.OUTPUT)
#



