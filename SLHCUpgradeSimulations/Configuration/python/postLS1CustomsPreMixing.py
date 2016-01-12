
import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.Configuration.muonCustomsPreMixing import customise_csc_PostLS1

def customisePostLS1(process):

    # deal with premixing-specific CSC changes separately
    process = customise_csc_PostLS1(process)

    return process


def customisePostLS1_50ns(process):

    # deal with premixing-specific CSC changes separately
    process = customise_csc_PostLS1(process)

    return process


def customisePostLS1_HI(process):

    # deal with premixing-specific CSC changes separately
    process = customise_csc_PostLS1(process)

    return process

