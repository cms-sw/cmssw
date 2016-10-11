import FWCore.ParameterSet.Config as cms

#GEN-SIM so far...
def customise(process):
    print "!!!You are using the SUPPORTED Flat version of the Phase2 Tracker !!!"
    n=0
    if hasattr(process,'reconstruction') or hasattr(process,'dqmoffline_step'):
        if hasattr(process,'mix'):
            if hasattr(process.mix,'input'):
                n=process.mix.input.nbPileupEvents.averageNumber.value()
        else:
            print 'phase1TkCustoms requires a --pileup option to cmsDriver to run the reconstruction/dqm'
            print 'Please provide one!'
            sys.exit(1)
    process=customise_condOverRides(process)

    return process

def customise_condOverRides(process):
    # this is a custom specific for tilted/flat .. how do we want to deal with it?
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkFlat_cff')
    return process

