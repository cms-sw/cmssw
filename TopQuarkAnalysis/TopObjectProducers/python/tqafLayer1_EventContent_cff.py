import FWCore.ParameterSet.Config as cms


def tqafLayer1EventContent(process):
    #
    # define event content
    #
    process.tqafEventContent = cms.PSet(
        outputCommands = cms.untracked.vstring('drop *')
    )
    #
    # tqaf layer1 event content equivalent to pat layer0 & 1
    #
    process.load("PhysicsTools.PatAlgos.patLayer1_EventContent_cff")
    process.tqafEventContent.outputCommands.extend(process.patLayer1EventContent.outputCommands)
    
    #
    # tqaf layer1 event content (on top of pat layer0 & 1)
    #
    process.patLayer1EventContent_tqaf = cms.PSet(
        outputCommands = cms.untracked.vstring('keep *_selectedLayer1CaloTaus_*_*',
                                               'keep *_decaySubset_*_*', 
                                               'keep *_initSubset_*_*', 
                                               'keep *_genEvt_*_*'
                                               )
    )
    process.tqafEventContent.outputCommands.extend(process.patLayer1EventContent_tqaf.outputCommands)
    
    #
    # pruned generator particles
    #
    process.include( "SimGeneral/HepPDTESSource/data/pythiapdt.cfi" )
    
    process.prunedGenParticles = cms.EDProducer(
        "GenParticlePruner",
        src = cms.InputTag("genParticles"),
        select = cms.vstring(
        ## keep relevant status 2 particles
        ##"keep+ pdgId = {t} && status = 2 || status == 3",
        ##"keep++ pdgId = {tbar} && status = 2 || status == 3",
        ## keep all stable particles within detector acceptance
        "keep status = 1 && pt > 0.5 && abs(eta) < 5"
        )
    )
    process.gen = cms.Path(process.prunedGenParticles)
    
    patLayer1EventContent_prunedGenParticles = cms.PSet(
        outputCommands = cms.untracked.vstring('drop *_genParticles_*_*',
                                               'keep *_prunedGenParticles_*_*'
                                               )
    )
    process.tqafEventContent.outputCommands.extend(patLayer1EventContent_prunedGenParticles.outputCommands)
        
    return()
