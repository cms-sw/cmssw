#
# This file contains the PF2PAT settings for the
# Top PAG reference selection for mu + jets analysis.
#

### Vertex configuration (for leptons)

pfVertices = 'goodOfflinePrimaryVertices'
pfD0Cut    = 0.2
pfDzCut    = 0.5

### Muon configuration

# Selector
pfMuonSelectionCut = 'pt > 5.'

# Isolation
pfMuonIsoConeR   = 0.4
pfMuonCombIsoCut = 0.15

### Electron configuration

# Selector
pfElectronSelectionCut  =     'pt > 5.'
pfElectronSelectionCut += ' && gsfTrackRef.isNonnull'
pfElectronSelectionCut += ' && gsfTrackRef.trackerExpectedHitsInner.numberOfLostHits < 2'

# Isolation
pfElectronIsoConeR   = 0.4
pfElectronCombIsoCut = 0.2
