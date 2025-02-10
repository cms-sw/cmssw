#include "SimG4Core/Notification/interface/TrackInformationExtractor.h"

#include "G4Track.hh"

const TrackInformation &TrackInformationExtractor::operator()(const G4Track &gtk) const {
  G4VUserTrackInformation *gui = gtk.GetUserInformation();
  const TrackInformation *tkInfo = dynamic_cast<const TrackInformation *>(gui);
  if (gui == nullptr) {
    missing(gtk);
  } else if (tkInfo == nullptr) {
    wrongType();
  }
  // Silence Clang analyzer warning: G4Exception will be thrown if tkInfo is null
  [[clang::suppress]] return *tkInfo;
}

TrackInformation &TrackInformationExtractor::operator()(G4Track &gtk) const {
  G4VUserTrackInformation *gui = gtk.GetUserInformation();
  TrackInformation *tkInfo = dynamic_cast<TrackInformation *>(gui);
  if (gui == nullptr) {
    missing(gtk);
  } else if (tkInfo == nullptr) {
    wrongType();
  }
  // Silence Clang analyzer warning: G4Exception will be thrown if tkInfo is null
  [[clang::suppress]] return *tkInfo;
}

void TrackInformationExtractor::missing(const G4Track &) const {
  G4Exception(
      "SimG4Core/Notification", "mc001", FatalException, "TrackInformationExtractor: G4Track has no TrackInformation");
}

void TrackInformationExtractor::wrongType() const {
  G4Exception(
      "SimG4Core/Notification", "mc001", FatalException, "User information in G4Track is not of TrackInformation type");
}
