import FWCore.ParameterSet.Config as cms

#GEN-SIM so far...
def customise(process):
    print "!!!You are using the SUPPORTED Tilted version of the Phase2 Tracker !!!"
    process=customise_condOverRides(process)
    return process

def customise_condOverRides(process):
    # this is a custom specific for tilted/flat .. how do we want to deal with it?
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkTilted4021_cff')
    return process

