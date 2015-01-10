
import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.Configuration.muonCustomsPreMixing import customise_csc_PostLS1,customise_csc_hlt
import postLS1Customs


def customisePostLS1(process):

    process = postLS1Customs.customisePostLS1(process)
    # deal with CSC separately:
    process = customise_csc_PostLS1(process)

    return process

