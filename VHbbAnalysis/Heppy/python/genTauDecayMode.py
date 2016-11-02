
#--------------------------------------------------------------------------------
# Auxiliary function to determine generator level tau decay mode.
#
# The function corresponds to a translation of
#    PhysicsTools/JetMCUtils/src/JetMCTag.cc
# to python.
#
# author: Christian Veelken, NICPB Tallinn
#--------------------------------------------------------------------------------

def genTauDecayMode(tau):
    
    numElectrons = 0
    numMuons = 0
    numChargedHadrons = 0
    numNeutralHadrons = 0
    numPhotons = 0

    daughters = tau.daughterPtrVector();
    for daughter in daughters:
        pdg_id = abs(daughter.pdgId())
        if pdg_id == 22:
            numPhotons = numPhotons + 1
        elif pdg_id == 11:
            numElectrons = numElectrons + 1
        elif pdg_id == 13:
            numMuons = numMuons + 1
        elif abs(daughter.charge()) > 0.5:
            numChargedHadrons = numChargedHadrons + 1
        else:
            numNeutralHadrons = numNeutralHadrons + 1

    ##print "<genTauDecayMode>: numChargedHadrons = %i, numNeutralHadrons = %i, numPhotons = %i" % (numChargedHadrons, numNeutralHadrons, numPhotons)

    if numElectrons == 1:
        return ( 0, "electron" )
    elif numMuons == 1:
        return ( 1, "muon" )
    elif numChargedHadrons == 1:
        if numNeutralHadrons != 0:
            return ( 5, "oneProngOther" )
        elif numPhotons == 0:
            return ( 2, "oneProng0Pi0" )
        elif numPhotons == 2:
            return ( 3, "oneProng1Pi0" )
        elif numPhotons == 4:
            return ( 4, "oneProng2Pi0" )
        else:
            return ( 5, "oneProngOther" )
    elif numChargedHadrons == 3:
        if numNeutralHadrons != 0:
            return ( 8, "threeProngOther" )
        elif numPhotons == 0:
            return ( 6, "threeProng0Pi0" )
        elif numPhotons == 2:
            return ( 7, "threeProng1Pi0" )
        else:
            return ( 8, "threeProngOther" )
    return ( 9, "rare" )
