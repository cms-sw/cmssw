import FWCore.ParameterSet.Config as cms

SimTauCPLink = cms.EDProducer("SimTauProducer",
                caloParticles      = cms.InputTag('mix', 'MergedCaloTruth'),
                genParticles      = cms.InputTag('genParticles'),
                genBarcodes       = cms.InputTag('genParticles')
                )
