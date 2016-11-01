import FWCore.ParameterSet.Config as cms


pregenerator = cms.EDProducer("EmbeddingProducer",
				   src = cms.InputTag("patMuonsAfterMediumID"),
				   vtxSrc = cms.InputTag(
				   "offlineSlimmedPrimaryVertices"
				   #"offlinePrimaryVertices"
				   ),
				   mixHepMc = cms.bool(False),
				   histFileName = cms.string("hist.root")
				  )
