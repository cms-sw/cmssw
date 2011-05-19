#
# This file contains the Top PAG reference selection for mu + jets analysis.
# as defined in
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopLeptonPlusJetsRefSel_mu#Selection_Version_SelV4_valid_fr
#

### Muon configuration

# PAT muons
muonsUsePV     = False # use beam spot rather than PV, which is necessary for 'dB' cut
muonEmbedTrack = True  # embedded track needed for muon ID cuts

### Jet configuration

# Jet algorithm
jetAlgo = 'AK5'

# JEC sets
jecSetBase = jetAlgo


### ------------------------- Reference selection -------------------------- ###


### Trigger selection

# HLT selection
#           run < 147196 (Run2010A)
triggerSelection_000000   = 'HLT_Mu9'
# 147196 <= run < 149442 (Run2010B)
triggerSelection_147196   = 'HLT_Mu15_v*'
# 160404 <= run < 163269 (Run2011A)
triggerSelection_160404   = 'HLT_Mu15_v* OR HLT_IsoMu17_v* OR HLT_Mu17_CentralJet30_v* OR HLT_Mu17_DiCentralJet30_v* OR HLT_Mu17_TriCentralJet30_v* OR HLT_Mu17_CentralJet30_BTagIP_v* OR HLT_IsoMu17_CentralJet30_BTagIP_v*'
# 163270 <= run < ...    (Run2011A)
triggerSelection_163270   = 'HLT_IsoMu17_v* OR HLT_Mu17_TriCentralJet30_v* OR HLT_Mu17_CentralJet30_BTagIP_v* OR HLT_IsoMu17_CentralJet30_BTagIP_v*' # un-prescaled only
triggerSelection_Summer11 = 'HLT_Mu20_v* OR HLT_Mu24_v* OR HLT_IsoMu17_v*'
triggerSelectionData = triggerSelection_163270
triggerSelectionMC   = triggerSelection_Summer11

### Muon selection

# Minimal selection for veto muons, also basis for signal muons
muonCutBase  =     'isGlobalMuon'   # general reconstruction property
muonCutBase += ' && pt > 10.'       # transverse momentum
muonCutBase += ' && abs(eta) < 2.5' # pseudo-rapisity range
# standard muon
muonCut  =  muonCutBase
muonCut += ' && (trackIso+caloIso)/pt < 0.2' # relative isolation
# PF muon
#muonCutPF  =  muonCutBase
muonCutPF  =  muonCut

# Signal muon selection on top of 'muonCut'
looseMuonCutBase  =     'isTrackerMuon'  # general reconstruction property
looseMuonCutBase += ' && pt > 20.'       # transverse momentum
looseMuonCutBase += ' && abs(eta) < 2.1' # pseudo-rapisity range
# standard muon
looseMuonCut  = looseMuonCutBase
looseMuonCut += ' && (trackIso+caloIso)/pt < 0.1'                           # relative isolation
looseMuonCut += ' && globalTrack.normalizedChi2 < 10.'                      # muon ID: 'isGlobalMuonPromptTight'
looseMuonCut += ' && globalTrack.hitPattern.numberOfValidMuonHits > 0'      # muon ID: 'isGlobalMuonPromptTight'
looseMuonCut += ' && abs(dB) < 0.02'                                        # 2-dim impact parameter with respect to beam spot (s. "PAT muon configuration" above)
looseMuonCut += ' && innerTrack.numberOfValidHits > 10'                     # tracker reconstruction
looseMuonCut += ' && innerTrack.hitPattern.pixelLayersWithMeasurement >= 1' # tracker reconstruction
looseMuonCut += ' && numberOfMatches > 1'                                   # muon chamber reconstruction
#looseMuonCut += ' && ()'                                                    # DeltaZ between muon vertex and PV < 1.
# PF muon
#looseMuonCutPF = looseMuonCutBase
looseMuonCutPF = looseMuonCut
# Signal muon distance from signal jet
muonJetsDR = 0.3                                                            # minimal DeltaR of signal muons from any signal jet

# Tightened signal muon selection on top of 'looseMuonCut'
tightMuonCutBase = ''
# standard muon
tightMuonCut  = tightMuonCutBase
tightMuonCut += '(trackIso+caloIso)/pt < 0.05' # relative isolation
# PF muon
#tightMuonCutPF  = tightMuonCutBase
tightMuonCutPF  = tightMuonCut

### Jet selection

# Signal jet selection
jetCutBase  =     'pt > 30.'            # transverse momentum
jetCutBase += ' && abs(eta) < 2.4'      # pseudo-rapisity range
# standard jet
jetCut  = jetCutBase
jetCut += ' && emEnergyFraction > 0.01' # jet ID: electro-magnetic energy fraction
jetCut += ' && jetID.n90Hits > 1'       # jet ID: number of RecHits carying 90% of the total energy
jetCut += ' && jetID.fHPD < 0.98'       # jet ID: fraction of energy in the hottest readout
# PF jet
jetCutPF  = jetCutBase
jetCutPF += ' && numberOfDaughters > 1'                                  # PF jet ID:
jetCutPF += ' && chargedEmEnergyFraction < 0.99'                         # PF jet ID:
jetCutPF += ' && neutralHadronEnergyFraction < 0.99'                     # PF jet ID:
jetCutPF += ' && neutralEmEnergyFraction < 0.99'                         # PF jet ID:
jetCutPF += ' && (chargedHadronEnergyFraction > 0. || abs(eta) >= 2.4)'  # PF jet ID:
jetCutPF += ' && (chargedMultiplicity > 0 || abs(eta) >= 2.4)'           # PF jet ID:
# Signal jet distance from signal muon
jetMuonsDRPF = 0.1

### Electron selection

# Veto electron selection
electronCutBase  =     'et > 15.'       # transverse energy
electronCutBase += ' && abs(eta) < 2.5' # pseudo-rapisity range
# standard electron
electronCut  = electronCutBase
electronCut += ' && (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/et < 0.2' # relative isolation
# PF electron
#electronCutPF  = electronCutBase
electronCutPF  = electronCut

### ------------------------------------------------------------------------ ###


### Trigger matching

# Trigger object selection
#           run < 147196 (Run2010A)
triggerObjectSelection_000000   = 'type("TriggerMuon") && ( path("HLT_Mu9") )'
# 147196 <= run < 149442 (Run2010B)
triggerObjectSelection_147196   = 'type("TriggerMuon") && ( path("HLT_Mu15_v*") )'
# 160404 <= run < 163269 (Run2011A)
triggerObjectSelection_160404   = 'type("TriggerMuon") && ( path("HLT_Mu15_v*") || path("HLT_IsoMu17_v*") || ( filter("hltL1Mu7CenJetL3MuFiltered17") && ( path("HLT_Mu17_CentralJet30_v*", 0) || path("HLT_Mu17_DiCentralJet30_v*", 0) || path("HLT_Mu17_TriCentralJet30_v*", 0) || path("HLT_Mu17_CentralJet30_BTagIP_v*", 0) ) ) || ( filter("hltIsoMu17CenJet30L3IsoFiltered17") && path("HLT_IsoMu17_CentralJet30_BTagIP_v*", 0) ) )'
# 163270 <= run < ...    (Run2011A)
triggerObjectSelection_163270   = 'type("TriggerMuon") && ( path("HLT_IsoMu17_v*") || ( filter("hltL1Mu7CenJetL3MuFiltered17") && ( path("HLT_Mu17_TriCentralJet30_v*", 0) || path("HLT_Mu17_CentralJet30_BTagIP_v*", 0) ) ) || ( filter("hltIsoMu17CenJet30L3IsoFiltered17") && path("HLT_IsoMu17_CentralJet30_BTagIP_v*", 0) ) )'
triggerObjectSelection_Summer11 = 'type("TriggerMuon") && ( path("HLT_Mu20_v*") || path("HLT_Mu24_v*") || path("HLT_IsoMu17_v*") )'
triggerObjectSelectionData = triggerObjectSelection_163270
triggerObjectSelectionMC   = triggerObjectSelection_Summer11
