#
# This file contains the Top PAG reference selection for mu + jets analysis.
# as defined in
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopLeptonPlusJetsRefSel_mu#Selection_Version_SelV4_valid_fr
#

# import the common object definitions
from TopQuarkAnalysis.Configuration.patRefSel_refMuJets import *

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
