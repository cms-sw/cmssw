#!/bin/bash

### Suggested by Piet.

echo "
# To save all contents
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_generatorSmeared_*_*')
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_ak4GenJets_*_*')
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_simCscTriggerPrimitiveDigis_*_*')


# Manual customization to switch off background hits
process.simMuonGEMDigis.digitizeOnlyMuons      = cms.bool(True)  # default: false
process.simMuonGEMDigis.doBkgNoise             = cms.bool(False) # default: true   
process.simMuonGEMDigis.doNoiseCLS             = cms.bool(False) # default: true
"
