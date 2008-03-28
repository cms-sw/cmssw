import FWCore.ParameterSet.Config as cms

#
# SUSYBSMAnalysis event content
# 
# the accumulated event contents should not be used, 
# but it's important that all EventContent definitions are included here
#
from SUSYBSMAnalysis.Skimming.SusyJetMET_EventContent_cff import *
from SUSYBSMAnalysis.Skimming.SusyElectronPhoton_EventContent_cff import *
from SUSYBSMAnalysis.Skimming.SusyMuon_EventContent_cff import *
#include "SUSYBSMAnalysis/Skimming/data/SusyJetMET_HLT_EventContent.cff"
#include "SUSYBSMAnalysis/Skimming/data/SusyElectronPhoton_HLT_EventContent.cff"
#include "SUSYBSMAnalysis/Skimming/data/SusyMuon_HLT_EventContent.cff"
from SUSYBSMAnalysis.Skimming.SusyMuonHits_EventContent_cff import *

