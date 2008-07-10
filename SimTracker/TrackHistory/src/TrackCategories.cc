
#include "SimTracker/TrackHistory/interface/TrackCategories.h"


static char * TrackCategoryName[] = {
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


std::ostream & operator<< (std::ostream & os, TrackCategories::Flags const & flags)
{
  bool init = true;

  // Print out the classification for the track
  for(std::size_t index = 0; index < flags.size(); ++index)
  {
    if (flags[index])
    {
      if (init)
      {
        os << TrackCategoryName[index];
        init = false;
      }
      else
        os << "::" << TrackCategoryName[index]; 
    }
  }
  os << std::endl;

  return os;
}


