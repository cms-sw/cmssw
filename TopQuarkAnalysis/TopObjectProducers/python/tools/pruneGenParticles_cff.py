import FWCore.ParameterSet.Config as cms

#----------------------------------------------------------------------------------------
#
#
# adds pruned genParticles to the tqafLayer1 output. Needs the following file(s) to be
# known before it can be executed:
#
#  * from TopQuarkAnalysis.TopObjectProducers.tqafLayer1_EventContent_cff   import *
#
#
#----------------------------------------------------------------------------------------

def tqafPruneGenParticles(process):
    process.include( "SimGeneral/HepPDTESSource/data/pythiapdt.cfi" )

## prune genParticles to what is needed for top analyses
    process.prunedGenParticles = cms.EDProducer(
      "GenParticlePruner",
      src = cms.InputTag("genParticles"),
      select = cms.vstring(
## keep all stable particles within detector acceptance
        "keep status = 1 && pt > 0.5 && abs(eta) < 5",
        "keep status = 2 && pdgId = {W+}",
        "keep status = 2 && pdgId = {W-}",
        "keep status = 2 && pdgId = {Z0}",
        "keep status = 3"
        )
    )
    process.gen = cms.Path(process.prunedGenParticles)

## replace genParticles by pruned genParticles
    tqafLayer1EventContent_genParticles = cms.PSet(
      outputCommands = cms.untracked.vstring(
        'drop *_genParticles_*_*',
        'keep *_prunedGenParticles_*_*'
      )
    )
    process.tqafEventContent.outputCommands.extend(tqafLayer1EventContent_genParticles.outputCommands)

    return()
