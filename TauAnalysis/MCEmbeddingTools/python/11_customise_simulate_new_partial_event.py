#import FWCore.ParameterSet.Config as cms
import os

# cuts: ---------------------------------------------------------------------

runOnTheGrid = False

productToUse =  'newSource'

#productToUse =  'newSourceWithMuons'
        
# ------------------------------------------------------------------------------
                        
def customise(process):
        process._Process__name="TAUSIMU"
        
        process.simSiPixelDigis.AddNoise = False
        process.simSiStripDigis.Noise = False
        #process.simEcalUnsuppressedDigis.doESNoise = False (*)
        process.simEcalUnsuppressedDigis.doNoise = False
        process.simHcalUnsuppressedDigis.doNoise = False
        process.simMuonCSCDigis.wires.doNoise = False
        process.simMuonRPCDigis.Noise = False
				
        process.load("Configuration.StandardSequences.Generator_cff")
				
        process.genParticles.src = productToUse
        process.genParticleCandidates.src = productToUse
        process.g4SimHits.Generator.HepMCProductLabel = productToUse
				
        if runOnTheGrid:
                process.source.fileNames=cms.untracked.vstring(__FILE_NAMES__)
                process.source.skipEvents=cms.untracked.uint32(__SKIP_EVENTS__)
                process.maxEvents.input = cms.untracked.int32(__MAX_EVENTS__)
                process.output.fileName = cms.untracked.string("output.root")
                
        process.output.outputCommands = cms.untracked.vstring(
          "drop *_*_*_*",
          "keep *_*_*_SELECTION",
          "keep *_*_*_TAUSIMU",
          "keep *_*_*_GEN",
          "keep edmHepMCProduct_*_*_*"
        )
        
        print process.schedule
        return(process)

