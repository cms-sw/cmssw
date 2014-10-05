
#include <iostream>

#include "SimTracker/TrackHistory/interface/TrackCategories.h"

const char * const TrackCategories::Names[] =
{
    "Fake",
    "Bad",
    "BadInnerHits",
    "SharedInnerHits",
    "SignalEvent",
    "Bottom",
    "Charm",
    "Light",
    "Muon",
    "TrackerSimHits",
    "BWeakDecay",
    "CWeakDecay",
    "ChargePionDecay",
    "ChargeKaonDecay",
    "TauDecay",
    "KsDecay",
    "LambdaDecay",
    "JpsiDecay",
    "XiDecay",
    "OmegaDecay",
    "SigmaPlusDecay",
    "SigmaMinusDecay",
    "LongLivedDecay",
    "KnownProcess",
    "UndefinedProcess",
    "UnknownProcess",
    "PrimaryProcess",
    "HadronicProcess",
    "DecayProcess",
    "ComptonProcess",
    "AnnihilationProcess",
    "EIoniProcess",
    "HIoniProcess",
    "MuIoniProcess",
    "PhotonProcess",
    "MuPairProdProcess",
    "ConversionsProcess",
    "EBremProcess",
    "SynchrotronRadiationProcess",
    "MuBremProcess",
    "MuNuclProcess",
    "FromBWeakDecayMuon",
    "FromCWeakDecayMuon",
    "DecayOnFlightMuon",
    "FromChargePionMuon",
    "FromChargeKaonMuon",
    "PrimaryVertex",
    "SecondaryVertex",
    "TertiaryVertex",
    "Unknown"
};


void TrackCategories::unknownTrack()
{
    // Check for all flags down
    for (std::size_t index = 0; index < flags_.size() - 1; ++index)
        if (flags_[index]) return;
    // If all of them are down then it is a unkown track.
    flags_[Unknown] = true;
}


std::ostream & operator<< (std::ostream & os, TrackCategories const & classifier)
{
    bool init = true;

    const TrackCategories::Flags & flags = classifier.flags();

    // Print out the classification for the track
    for (std::size_t index = 0; index < flags.size(); ++index)
    {
        if (flags[index])
        {
            if (init)
            {
                os << TrackCategories::Names[index];
                init = false;
            }
            else
                os << "::" << TrackCategories::Names[index];
        }
    }
    os << std::endl;

    return os;
}
