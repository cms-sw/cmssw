import FWCore.ParameterSet.Config as cms

##from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

def tqafLayer1GenParticles(process):
    
    process.include( "SimGeneral/HepPDTESSource/data/pythiapdt.cfi" )

    ## prune genParticles to what is needed for top analyses
    process.tqafGenParticles = cms.EDProducer(
        "GenParticlePruner",
        src = cms.InputTag("genParticles"),
        select = cms.vstring(
        ## keep all top daughters relevant for TopGenEvent
        ##"keep++ pdgId = {t} && status = 2 || status == 3",
        ##"keep++ pdgId = {tbar} && status = 2 || status == 3",
        ## keep all stable particles within detector acceptance
        "keep status = 1 && pt > 1.5 && abs(eta) < 5"
        )
    )
    process.gen = cms.Path(process.tqafGenParticles)
    
    ## replace genParticles by pruned genParticles
    tqafLayer1EventContent_genParticles = cms.PSet(
        outputCommands = cms.untracked.vstring(
        'drop *_genParticles_*_*',
        'keep *_tqafGenParticles_*_*'
        )
    )
    process.tqafEventContent.outputCommands.extend(tqafLayer1EventContent_genParticles.outputCommands)

    return()
