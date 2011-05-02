#
# This file contains the Top PAG reference selection for mu + jets analysis
# the documentation can be found in
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopLeptonPlusJetsRefSel_mu
#

### Muon configuration

# PAT muons
muonsUsePV     = False # use beam spot rather than PV, which is necessary for 'dB' cut
muonEmbedTrack = True  # embedded track needed for muon ID cuts

### ------------------------- Reference selection -------------------------- ###

### Trigger selection

# HLT selection
triggerSelection = 'HLT_Mu15 OR HLT_Mu15_v*'

### Muon selection

# Minimal selection for veto muons
muonCut  =     'isGlobalMuon'                # general reconstruction property
muonCut += ' && pt > 10.'                    # transverse momentum
muonCut += ' && abs(eta) < 2.5'              # pseudo-rapisity range
muonCut += ' && (trackIso+caloIso)/pt < 0.2' # relative isolation

# Signal muon selection on top of 'muonCut'
looseMuonCut  =     'isTrackerMuon'                                         # general reconstruction property
looseMuonCut += ' && pt > 20.'                                              # transverse momentum
looseMuonCut += ' && abs(eta) < 2.1'                                        # pseudo-rapisity range
looseMuonCut += ' && (trackIso+caloIso)/pt < 0.1'                           # relative isolation
looseMuonCut += ' && globalTrack.normalizedChi2 < 10.'                      # muon ID: 'isGlobalMuonPromptTight'
looseMuonCut += ' && globalTrack.hitPattern.numberOfValidMuonHits > 0'      # muon ID: 'isGlobalMuonPromptTight'
looseMuonCut += ' && abs(dB) < 0.02'                                        # 2-dim impact parameter with respect to beam spot (s. "PAT muon configuration" above)
looseMuonCut += ' && innerTrack.numberOfValidHits > 10'                     # tracker reconstruction
looseMuonCut += ' && innerTrack.hitPattern.pixelLayersWithMeasurement >= 1' # tracker reconstruction
looseMuonCut += ' && numberOfMatches > 1'                                   # muon chamber reconstruction
#looseMuonCut += ' && ()'                                                    # DeltaZ between muon vertex and PV < 1.
muonJetsDR = 0.3                                                            # minimal DeltaR of signal muons from any signal jet

# Tightened signal muon selection on top of 'looseMuonCut'
tightMuonCut  = '(trackIso+caloIso)/pt < 0.05' # relative isolation

### Jet selection

# Signal jet selection
jetCut  =     'pt > 30.'
jetCut += ' && abs(eta) < 2.4'          # transverse momentum# pseudo-rapisity range
jetCut += ' && emEnergyFraction > 0.01' # jet ID: electro-magnetic energy fraction
jetCut += ' && jetID.n90Hits > 1'       # jet ID: number of RecHits carying 90% of the total energy
jetCut += ' && jetID.fHPD < 0.98'       # jet ID: fraction of energy in the hottest readout

### Electron selection

# Veto electron selection
electronCut  =     'et > 15.'                                                      # transverse momentum
electronCut += ' && abs(eta) < 2.5'                                                # pseudo-rapisity range
electronCut += ' && (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/et < 0.2' # relative isolation

### ------------------------------------------------------------------------ ###

### Trigger matching

# Trigger object selection
triggerObjectSelection = 'type("TriggerMuon") && ( path("HLT_Mu15") || path("HLT_Mu15_v*") )' # run >= 147196
