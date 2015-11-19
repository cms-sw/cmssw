#!/bin/bash

### Suggested by Piet.

echo "
# Manual customization to switch off background hits
process.simMuonGEMDigis.digitizeOnlyMuons      = cms.bool(True)  # default: false
process.simMuonGEMDigis.doBkgNoise             = cms.bool(False) # default: true   
process.simMuonGEMDigis.doNoiseCLS             = cms.bool(False) # default: true
"
