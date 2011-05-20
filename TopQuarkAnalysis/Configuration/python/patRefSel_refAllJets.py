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

#           run < 147196 (Run2010A)
# 147196 <= run < 149442 (Run2010B)
# 160404 <= run < 163869 (Run2011A)
triggerSelection_160404 = 'HLT_QuadJet50_Jet40_v*'  # run < 147196
# 163869 <= run < ...    (Run2011A)
# Summer11 MC
triggerSelection_Summer11= 'HLT_QuadJet50_Jet40_v*'  # run < 147196

triggerSelectionData = triggerSelection_160404
triggerSelectionMC   = triggerSelection_Summer11

### Jet selection

jetCutMedium = ' && pt > 50.'
jetCutHard   = ' && pt > 60.'

### Trigger matching

# Trigger object selection
#           run < 147196 (Run2010A)
# 147196 <= run < 149442 (Run2010B)
# 160404 <= run < 163869 (Run2011A)
triggerObjectSelection_160404   = 'type("TriggerJet") && ( path("HLT_QuadJet50_Jet40_v*") )'
# 163869 <= run < ...    (Run2011A)
# Summer11 MC
triggerObjectSelection_Summer11 = 'type("TriggerJet") && ( path("HLT_QuadJet50_Jet40_v*") )'

triggerObjectSelectionData = triggerObjectSelection_160404
triggerObjectSelectionMC   = triggerObjectSelection_Summer11
