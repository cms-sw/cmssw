#
# This file contains the PF2PAT settings for the Top PAG reference selections.
#

### Vertex configuration (for leptons)

pfVertices  = 'goodOfflinePrimaryVertices'
pfD0Cut     = 0.2
pfDzCut     = 0.5

### Muon configuration

# Selector
pfMuonSelectionCut = 'pt > 5.'

# Isolation
pfMuonIsoConeR03 = False
pfMuonCombIsoCut = 0.2

### Electron configuration

# Selector
pfElectronSelectionCut  =     'pt > 5.'
pfElectronSelectionCut += ' && gsfTrackRef.isNonnull'
pfElectronSelectionCut += ' && gsfTrackRef.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\') < 2'

# Isolation
pfElectronIsoConeR03 = True
pfElectronCombIsoCut = 0.2
