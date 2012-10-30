
import FWCore.ParameterSet.Config as cms

from muonCustoms import customise_csc_geom_cond_digi
from customise_mixing import customise_NoCrossing

def digiCustoms(process):
#    process=customise_NoCrossing(process)
    process=customise_csc_geom_cond_digi(process)
    return process
