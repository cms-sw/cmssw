#include "SimG4Core/Notification/interface/TrackInformation.h"

#include <iostream>

using std::cout;
using std::endl;

G4Allocator<TrackInformation> TrackInformationAllocator;

void TrackInformation::Print() const
{
    cout << " TrackInformation : storeTrack = " << storeTrack_ << endl;
    cout << " TrackInformation : hasHits = "    << hasHits_    << endl;
    cout << " TrackInformation : isPrimary = "  << isPrimary_  << endl;
    cout << " TrackInformation : isGeneratedSecondary = "  << isGeneratedSecondary_ << endl;
    cout << " TrackInformation : isInHistory = "  << isInHistory_ << endl;
    cout << " TrackInformation : IDonCaloSurface = "  << getIDonCaloSurface() << endl;
    cout << " TrackInformation : caloIDChecked = "  << caloIDChecked() << endl;
}

