
import FWCore.ParameterSet.Config as cms

from muon_customs import customise_csc_geom_cond_digi

def postLS1Customs(process):
    process=customise_csc_geom_cond_digi(process)
    return process
