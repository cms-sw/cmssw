
#include <iostream>

#include "SimTracker/TrackHistory/interface/VertexCategories.h"

const char * VertexCategories::Names[] =
{
    "Fake",
    "SignalEvent",
    "BWeakDecay",
    "CWeakDecay",
    "TauDecay",
    "KsDecay",
    "LambdaDecay",
    "LongLivedDecay",
    "Conversion",
    "Interaction",
    "PrimaryVertex",
    "SecondaryVertex",
    "TertiaryVertex",
    "Unknown"
};


std::ostream & operator<< (std::ostream & os, VertexCategories const & classifier)
{
    bool init = true;

    const VertexCategories::Flags & flags = classifier.flags();

    // Print out the classification for the track
    for (std::size_t index = 0; index < flags.size(); ++index)
    {
        if (flags[index])
        {
            if (init)
            {
                os << VertexCategories::Names[index];
                init = false;
            }
            else
                os << "::" << VertexCategories::Names[index];
        }
    }
    os << std::endl;

    return os;
}
