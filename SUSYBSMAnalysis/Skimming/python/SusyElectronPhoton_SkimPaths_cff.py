import FWCore.ParameterSet.Config as cms

#TO BE RUN ON: PDElectron and PDPhoton
#   WG1
from SUSYBSMAnalysis.CSA07Skims.lepSUSY_0Muon_2Elec_2Jets_MET_Path_cff import *
from SUSYBSMAnalysis.CSA07Skims.lepSUSY_0Muon_1Elec_1Jets_MET_Path_cff import *
from SUSYBSMAnalysis.CSA07Skims.lepSUSY_1Muon_1Elec_2Jets_MET_Path_cff import *
from SUSYBSMAnalysis.CSA07Skims.lepSUSY_0Muon_1Elec_2Jets_MET_Path_cff import *
#include "SUSYBSMAnalysis/CSA07Skims/data/lepSUSY_1Muon_0Elec_1Jets_MET_Path.cff"
#include "SUSYBSMAnalysis/CSA07Skims/data/lepSUSY_1Muon_0Elec_2Jets_MET_Path.cff"
#include "SUSYBSMAnalysis/CSA07Skims/data/lepSUSY_2Muon_0Elec_2Jets_MET_Path.cff"
#   WG2
from SUSYBSMAnalysis.CSA07Skims.hadSUSYdiElecPath_cff import *
from SUSYBSMAnalysis.CSA07Skims.hadSUSYTopElecPath_cff import *
#  WG3
from SUSYBSMAnalysis.CSA07Skims.SUSYHighPtPhotonPath_cff import *
from SUSYBSMAnalysis.CSA07Skims.SUSYControlHighPtPhotonPath_cff import *
#   WG4 HEEP Paths
from SUSYBSMAnalysis.CSA07Skims.HEEPSignalHighEtPath_cff import *
#FIXME:  commented because of missing EtMinSuperClusterCountFilter
from SUSYBSMAnalysis.CSA07Skims.HEEPSignalMedEtPath_cff import *

