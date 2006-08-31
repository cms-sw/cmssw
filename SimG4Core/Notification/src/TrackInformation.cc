#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

using std::cout;
using std::endl;

G4Allocator<TrackInformation> TrackInformationAllocator;

void TrackInformation::Print() const
{
     LogDebug("TrackInformation") << " TrackInformation : storeTrack = " << storeTrack_;
     LogDebug("TrackInformation") << " TrackInformation : hasHits = "    << hasHits_;
     LogDebug("TrackInformation") << " TrackInformation : isPrimary = "  << isPrimary_;
     LogDebug("TrackInformation") << " TrackInformation : isGeneratedSecondary = "  << isGeneratedSecondary_;
     LogDebug("TrackInformation") << " TrackInformation : isInHistory = "  << isInHistory_;
     LogDebug("TrackInformation") << " TrackInformation : IDonCaloSurface = "  << getIDonCaloSurface();
     LogDebug("TrackInformation") << " TrackInformation : caloIDChecked = "  << caloIDChecked();
}

