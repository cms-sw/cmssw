
#include <iostream>

#include "SimTracker/TrackHistory/interface/TrackCategories.h"

const char * TrackCategories::Names[] =
{
    "Fake",
    "Bad",
    "SignalEvent",
    "Bottom",
    "Charm",
    "Light",
    "BWeakDecay",
    "CWeakDecay",
    "TauDecay",
    "KsDecay",
    "LambdaDecay",
    "Jpsi",
    "Xi",
    "Omega",
    "SigmaPlus",
    "SigmaMinus",
    "LongLivedDecay",
    "Conversion",
    "Interaction",
    "PrimaryVertex",
    "SecondaryVertex",
    "TertiaryVertex",
    "BadInnerHits",
    "SharedInnerHits",
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
