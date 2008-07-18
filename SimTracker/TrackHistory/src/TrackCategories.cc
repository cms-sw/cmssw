
#include "SimTracker/TrackHistory/interface/TrackCategories.h"

const char * TrackCategories::Names[] = {
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
  "LongLivedDecay",
  "Conversion",
  "Interaction",
  "PrimaryVertex",
  "SecondaryVertex",
  "TierciaryVertex",
  "Unknown"
};

/*TrackCategories::TrackCategories()
{
  names.reserve(TrackCategories::Unknown+1);

  names.push_back("Fake");
names.push_back("Bad");
names.push_back("SignalEvent");
names.push_back("Bottom");
names.push_back("Charm");
names.push_back("Light");
names.push_back("BWeakDecay");
names.push_back("CWeakDecay");
names.push_back("TauDecay");
names.push_back("KsDecay");
names.push_back("LambdaDecay");
names.push_back("LongLivedDecay");
names.push_back("Conversion");
names.push_back("Interaction");
names.push_back("PrimaryVertex");
names.push_back("SecondaryVertex");
names.push_back("TierciaryVertex");
names.push_back("Unknown");

}*/

std::ostream & operator<< (std::ostream & os, TrackCategories::Flags const & flags)
{
    bool init = true;

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


