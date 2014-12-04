#
# This file contains the Top PAG reference selection for mu + jets analysis.
#


### ------------------------- Reference selection -------------------------- ###


### Trigger selection

# HLT selection
triggerSelectionData  = 'HLT_IsoMu24_eta2p1_v*'
triggerSelectionMC    = 'HLT_*' # not recommended

### Muon selection

# Minimal selection for all muons, also basis for signal and veto muons
# Muon ID ("loose")
muonCut  =     'isPFMuon'                                                                      # general reconstruction property
muonCut += ' && (isGlobalMuon || isTrackerMuon)'                                               # general reconstruction property
# Kinematics
muonCut += ' && pt > 10.'                                                                      # transverse momentum
muonCut += ' && abs(eta) < 2.5'                                                                # pseudo-rapisity range
# (Relative) isolation
muonCut += ' && (chargedHadronIso+neutralHadronIso+photonIso-0.5*puChargedHadronIso)/pt < 0.2' # relative isolation w/ Delta beta corrections (factor 0.5)

# Signal muon selection on top of 'muonCut'
# Muon ID ("tight")
signalMuonCut  =     'isPFMuon'                                                                               # general reconstruction property
signalMuonCut += ' && isGlobalMuon'                                                                           # general reconstruction property
signalMuonCut += ' && globalTrack.normalizedChi2 < 10.'                                                       # muon ID: 'isGlobalMuonPromptTight'
signalMuonCut += ' && track.hitPattern.trackerLayersWithMeasurement > 5'                                      # muon ID: 'isGlobalMuonPromptTight'
signalMuonCut += ' && globalTrack.hitPattern.numberOfValidMuonHits > 0'                                       # muon ID: 'isGlobalMuonPromptTight'
signalMuonCut += ' && abs(dB) < 0.2'                                                                          # 2-dim impact parameter with respect to beam spot (s. "PAT muon configuration" above)
signalMuonCut += ' && innerTrack.hitPattern.numberOfValidPixelHits > 0'                                       # tracker reconstruction
signalMuonCut += ' && numberOfMatchedStations > 1'                                                            # muon chamber reconstruction
# Kinematics
signalMuonCut += ' && pt > 26.'                                                                               # transverse momentum
signalMuonCut += ' && abs(eta) < 2.1'                                                                         # pseudo-rapisity range
# (Relative) isolation
signalMuonCut += ' && (chargedHadronIso+max(0.,neutralHadronIso+photonIso-0.5*puChargedHadronIso))/pt < 0.12' # relative isolation w/ Delta beta corrections (factor 0.5)

muonVertexMaxDZ = 0.5 # DeltaZ between muon vertex and PV

### Jet selection

# Signal jet selection
# Jet ID
jetCut  =     'numberOfDaughters > 1'                                 # PF jet ID:
jetCut += ' && neutralHadronEnergyFraction < 0.99'                    # PF jet ID:
jetCut += ' && neutralEmEnergyFraction < 0.99'                        # PF jet ID:
jetCut += ' && (chargedEmEnergyFraction < 0.99 || abs(eta) >= 2.4)'   # PF jet ID:
jetCut += ' && (chargedHadronEnergyFraction > 0. || abs(eta) >= 2.4)' # PF jet ID:
jetCut += ' && (chargedMultiplicity > 0 || abs(eta) >= 2.4)'          # PF jet ID:
# Kinematics
jetCut  +' && abs(eta) < 2.5'                                        # pseudo-rapisity range
# varying jet pt thresholds
veryLooseJetCut = 'pt > 30.' # transverse momentum (4 jets)
looseJetCut     = 'pt > 30.' # transverse momentum (3 jets)
tightJetCut     = 'pt > 30.' # transverse momentum (2 jets)
veryTightJetCut = 'pt > 30.' # transverse momentum (leading jet)

### Electron selection

# Minimal selection for veto electrons
# ... using GsfElectron kinematics
# Electron ID
electronGsfCut  =     'electronID("cutBasedElectronID-CSA14-50ns-V1-standalone-veto")'                                                  # electrons ID
# Kinematics
electronGsfCut += ' && ecalDrivenMomentum.pt > 20.'                                                                                     # transverse energy
electronGsfCut += ' && abs(ecalDrivenMomentum.eta) < 2.5'                                                                               # pseudo-rapisity range
# (Relative) isolation
electronGsfCut += ' && (chargedHadronIso+max(0.,neutralHadronIso+photonIso-1.0*userIsolation("User1Iso")))/ecalDrivenMomentum.pt < 0.2' # relative isolation with Delta beta corrections
# ... using re-calibrated (with regression energy) kinematics
electronCalibCut = electronGsfCut.replace( 'ecalDrivenMomentum.', '' )
electronCut = electronGsfCut
### ------------------------------------------------------------------------ ###

### Electron selection

# Signal b-tagged jet selection
bTagCut = 'bDiscriminator("combinedInclusiveSecondaryVertexV2BJetTags") > 0.679'


### Trigger matching

# Trigger object selection
triggerObjectSelectionData = 'type("TriggerMuon") && ( path("%s") )'%( triggerSelectionData )
triggerObjectSelectionMC   = 'type("TriggerMuon") && ( path("%s") )'%( triggerSelectionMC )
