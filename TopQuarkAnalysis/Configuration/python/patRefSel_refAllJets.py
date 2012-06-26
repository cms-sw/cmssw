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


# PF2PAT settings
from TopQuarkAnalysis.Configuration.patRefSel_PF2PAT import *

### Trigger selection

# HLT selection
triggerSelectionDataRelVals = 'HLT_IsoMu17_eta2p1_TriCentralJet30_v*' # 2011B RelVals
triggerSelectionData        = 'HLT_Iso10Mu20_eta2p1_TriCentralPFJet30_v*'
triggerSelectionMC          = 'HLT_IsoMu20_eta2p1_TriCentralPFJet30_v*'

### Muon selection

# Minimal selection for all muons, also basis for signal and veto muons
muonCutBase  =     'pt > 10.'                                                    # transverse momentum
muonCutBase += ' && abs(eta) < 2.5'                                              # pseudo-rapisity range
# veto muon
muonCutVeto  = muonCutBase
muonCutVeto += ' && isGlobalMuon'                                                # general reconstruction property
# standard muon
muonCut  = muonCutVeto
muonCut += ' && (trackIso+caloIso)/pt < 0.2'                                     # relative isolation
# PF muon
muonCutPF  =  muonCutVeto
muonCutPF += ' && (chargedHadronIso+neutralHadronIso+photonIso)/pt < 0.2'        # relative isolation

# Signal muon selection on top of 'muonCut'
looseMuonCutBase  =     'isTrackerMuon'                                           # general reconstruction property
looseMuonCutBase += ' && pt > 20.'                                                # transverse momentum
looseMuonCutBase += ' && abs(eta) < 2.1'                                          # pseudo-rapisity range
looseMuonCutBase += ' && globalTrack.normalizedChi2 < 10.'                        # muon ID: 'isGlobalMuonPromptTight'
looseMuonCutBase += ' && globalTrack.hitPattern.trackerLayersWithMeasurement > 5' # muon ID: 'isGlobalMuonPromptTight'
looseMuonCutBase += ' && globalTrack.hitPattern.numberOfValidMuonHits > 0'        # muon ID: 'isGlobalMuonPromptTight'
looseMuonCutBase += ' && abs(dB) < 0.02'                                          # 2-dim impact parameter with respect to beam spot (s. "PAT muon configuration" above)
looseMuonCutBase += ' && innerTrack.numberOfValidHits > 10'                       # tracker reconstruction
looseMuonCutBase += ' && innerTrack.hitPattern.numberOfValidPixelHits > 0'        # tracker reconstruction
looseMuonCutBase += ' && numberOfMatchedStations > 1'                             # muon chamber reconstruction
#looseMuonCut += ' && ...'                                                         # DeltaZ between muon vertex and PV < 1. not accessible via configuration yet
# standard muon
looseMuonCut  = looseMuonCutBase
looseMuonCut += ' && (trackIso+caloIso)/pt < 0.1'                                # relative isolation
# PF muon
looseMuonCutPF = looseMuonCutBase
looseMuonCutPF += ' && (chargedHadronIso+neutralHadronIso+photonIso)/pt < 0.125' # relative isolation
# Signal muon distance from signal jet
muonJetsDR = 0.3                                                                 # minimal DeltaR of signal muons from any signal jet

# Tightened signal muon selection on top of 'looseMuonCut'
tightMuonCutBase  =     ''
# standard muon
tightMuonCut  = tightMuonCutBase
tightMuonCut += '(trackIso+caloIso)/pt < 0.05'                                   # relative isolation
# PF muon
tightMuonCutPF  = tightMuonCutBase
tightMuonCutPF += '(chargedHadronIso+neutralHadronIso+photonIso)/pt < 0.125'     # relative isolation

### Jet selection

# Signal jet selection
jetCutBase  =     'pt > 35.'                                             # transverse momentum
jetCutBase += ' && abs(eta) < 2.5'                                       # pseudo-rapisity range
# standard jet
jetCut  = jetCutBase
jetCut += ' && emEnergyFraction > 0.01'                                  # jet ID: electro-magnetic energy fraction
jetCut += ' && jetID.n90Hits > 1'                                        # jet ID: number of RecHits carying 90% of the total energy
jetCut += ' && jetID.fHPD < 0.98'                                        # jet ID: fraction of energy in the hottest readout
# PF jet
jetCutPF  = jetCutBase
jetCutPF += ' && numberOfDaughters > 1'                                  # PF jet ID:
jetCutPF += ' && neutralHadronEnergyFraction < 0.99'                     # PF jet ID:
jetCutPF += ' && neutralEmEnergyFraction < 0.99'                         # PF jet ID:
jetCutPF += ' && (chargedEmEnergyFraction < 0.99 || abs(eta) >= 2.4)'    # PF jet ID:
jetCutPF += ' && (chargedHadronEnergyFraction > 0. || abs(eta) >= 2.4)'  # PF jet ID:
jetCutPF += ' && (chargedMultiplicity > 0 || abs(eta) >= 2.4)'           # PF jet ID:
# Signal jet distance from signal muon
jetMuonsDRPF = 0.1

### Electron selection

# Minimal selection for all electrons, also basis for signal and veto muons
electronCutBase  =     'pt > 20.'                                                  # transverse energy
electronCutBase += ' && abs(eta) < 2.5'                                            # pseudo-rapisity range
# veto electron
electronCutVeto  = electronCutBase
electronCutVeto += ' && electronID("mvaTrigV0") > 0.'                              # MVA electrons ID
# standard electron
electronCut  = electronCutVeto
electronCut += ' && (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/et < 0.2' # relative isolation
# PF electron
electronCutPF  = electronCutVeto
electronCutPF += ' && (chargedHadronIso+max(0.,neutralHadronIso)+photonIso-0.5*puChargedHadronIso)/et < 0.2' # relative isolation with Delta beta corrections

### ------------------------------------------------------------------------ ###


### Trigger matching

# Trigger object selection
triggerObjectSelectionDataRelVals = 'type("TriggerMuon") && ( path("HLT_IsoMu17_eta2p1_TriCentralJet30_v*") )' # 2011B RelVals
triggerObjectSelectionData        = 'type("TriggerMuon") && ( path("HLT_Iso10Mu20_eta2p1_TriCentralPFJet30_v*") )'
triggerObjectSelectionMC          = 'type("TriggerMuon") && ( path("HLT_IsoMu20_eta2p1_TriCentralPFJet30_v*") )'
## special replacements for the analysis can be done here


### Trigger selection

# HLT selection

# Trigger selection
triggerSelectionDataRelVals = 'HLT_QuadJet50_DiJet40_v*' # 2011B RelVals
triggerSelectionData        = 'HLT_*' # not defined yet
triggerSelectionMC          = 'HLT_QuadPFJet75_55_35_20_BTagCSV_VBF_v*' # not defined yet

### Jet selection

jetCutMedium = ' && pt > 50.'
jetCutHard   = ' && pt > 60.'

### Trigger matching

# Trigger object selection
triggerObjectSelectionDataRelVals = 'type("TriggerJet") && ( path("HLT_QuadJet50_DiJet40_v*") )' # 2011B RelVals
triggerObjectSelectionData        = 'type("TriggerJet") && ( path("HLT_*") )' # not defined yet
triggerObjectSelectionMC          = 'type("TriggerJet") && ( path("HLT_QuadPFJet75_55_35_20_BTagCSV_VBF_v*") )' # not defined yet
