#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

G4Allocator<TrackInformation> TrackInformationAllocator;

void TrackInformation::Print() const {
  LogDebug("TrackInformation") << " TrackInformation : storeTrack = " << storeTrack_ << "\n"
			       << "                    hasHits = "    << hasHits_ << "\n"
			       << "                    isPrimary = "  << isPrimary_ << "\n"
			       << "                    isGeneratedSecondary = "  << isGeneratedSecondary_ << "\n"
			       << "                    isInHistory = "  << isInHistory_ << "\n"
			       << "                    idOnCaloSurface = "  << getIDonCaloSurface() << "\n"
			       << "                    caloIDChecked = "  << caloIDChecked() << "\n"
			       << "                    idCaloVolume = " << idCaloVolume_ << "\n"
			       << "                    idLastVolume = " << idLastVolume_;
}

