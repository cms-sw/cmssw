
import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.Configuration.muonCustomsPreMixing import customise_csc_PostLS1
import postLS1Customs


def customisePostLS1(process):

    # apply the general 25 ns post-LS1 customisation
    process = postLS1Customs.customisePostLS1(process)
    # deal with premixing-specific CSC changes separately
    process = customise_csc_PostLS1(process)

    return process


def customisePostLS1_50ns(process):

    # apply the general 25 ns post-LS1 customisation
    process = postLS1Customs.customisePostLS1_50ns(process)
    # deal with premixing-specific CSC changes separately
    process = customise_csc_PostLS1(process)

    return process


def customisePostLS1_HI(process):

    # apply the general 25 ns post-LS1 customisation
    process = postLS1Customs.customisePostLS1_HI(process)
    # deal with premixing-specific CSC changes separately
    process = customise_csc_PostLS1(process)

    return process

