import FWCore.ParameterSet.Config as cms

def tpInM3P1BXRange(process) :

    if hasattr(process,"mix"):
        
        process.mix.digitizers.mergedtruth.maximumPreviousBunchCrossing   = 3
        process.mix.digitizers.mergedtruth.maximumSubsequentBunchCrossing = 1

    return process


def signalOnlyTP(process) :

    if hasattr(process,"mix"):
        
        process.mix.digitizers.mergedtruth.select.signalOnlyTP = True

    return process


def inTimeOnlyTP(process) :

    if hasattr(process,"mix"):
        
        process.mix.digitizers.mergedtruth.select.intimeOnlyTP = True

    return process

def higherPtTP(process) :

    if hasattr(process,"mix"):
        
        process.mix.digitizers.mergedtruth.select.ptMinTP = 1.

    return process


    
    
        

