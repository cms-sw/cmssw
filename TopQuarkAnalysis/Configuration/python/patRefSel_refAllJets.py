#
# This file contains the Top PAG reference selection for mu + jets analysis.
# as defined in
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopLeptonPlusJetsRefSel_mu#Selection_Version_SelV4_valid_fr
#

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
triggerSelectionDataRelVals = 'HLT_QuadJet50_DiJet40_v*' # 2011B RelVals
triggerSelectionData        = 'HLT_*' # not defined yet
triggerSelectionMC          = 'HLT_QuadJet60_DiJet20_v*' # not defined yet

### Muon selection

# Minimal selection for all muons, also basis for signal and veto muons
muonCut  =     'isPFMuon'                                                                      # general reconstruction property
muonCut += ' && (isGlobalMuon || isTrackerMuon)'                                               # general reconstruction property
muonCut += ' && pt > 10.'                                                                      # transverse momentum
muonCut += ' && abs(eta) < 2.5'                                                                # pseudo-rapisity range
muonCut += ' && (chargedHadronIso+neutralHadronIso+photonIso-0.5*puChargedHadronIso)/pt < 0.2' # relative isolation w/ Delta beta corrections (factor 0.5)

### Jet selection

# Signal jet selection
jetCut  =     'pt > 20.'                                              # transverse momentum
jetCut += ' && abs(eta) < 2.5'                                        # pseudo-rapisity range
jetCut += ' && numberOfDaughters > 1'                                 # PF jet ID:
jetCut += ' && neutralHadronEnergyFraction < 0.99'                    # PF jet ID:
jetCut += ' && neutralEmEnergyFraction < 0.99'                        # PF jet ID:
jetCut += ' && (chargedEmEnergyFraction < 0.99 || abs(eta) >= 2.4)'   # PF jet ID:
jetCut += ' && (chargedHadronEnergyFraction > 0. || abs(eta) >= 2.4)' # PF jet ID:
jetCut += ' && (chargedMultiplicity > 0 || abs(eta) >= 2.4)'          # PF jet ID:
# varying jet pt thresholds
veryLooseJetCut = 'pt > 35.' # transverse momentum (all jets)
looseJetCut     = 'pt > 50.' # transverse momentum (3rd jet, optional)
tightJetCut     = 'pt > 60.' # transverse momentum (leading jets)

### Electron selection

# Minimal selection for all electrons, also basis for signal and veto muons
electronCut  =     'pt > 20.'                                                                              # transverse energy
electronCut += ' && abs(eta) < 2.5'                                                                        # pseudo-rapisity range
electronCut += ' && electronID("mvaTrigV0") > 0.'                                                          # MVA electrons ID
electronCut += ' && (chargedHadronIso+max(0.,neutralHadronIso)+photonIso-0.5*puChargedHadronIso)/et < 0.2' # relative isolation with Delta beta corrections

### ------------------------------------------------------------------------ ###


### Trigger matching

# Trigger object selection
triggerObjectSelectionDataRelVals = 'type("TriggerJet") && ( path("HLT_QuadJet50_DiJet40_v*") )' # 2011B RelVals
triggerObjectSelectionData        = 'type("TriggerJet") && ( path("HLT_*") )' # not defined yet
triggerObjectSelectionMC          = 'type("TriggerJet") && ( path("HLT_QuadPFJet75_55_35_20_BTagCSV_VBF_v*") )' # not defined yet
