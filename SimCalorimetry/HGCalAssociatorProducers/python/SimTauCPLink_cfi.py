import FWCore.ParameterSet.Config as cms

SimTauCPLink = cms.EDProducer("SimTauProducer",
                CaloParticle      = cms.InputTag('mix', 'MergedCaloTruth'),
                GenParticles      = cms.InputTag('genParticles'),
                GenBarcodes       = cms.InputTag('genParticles')
                )
