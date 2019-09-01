#include "SimG4Core/Notification/interface/TrackInformationExtractor.h"

#include "G4Track.hh"

const TrackInformation &TrackInformationExtractor::operator()(const G4Track &gtk) const {
  G4VUserTrackInformation *gui = gtk.GetUserInformation();
  if (gui == nullptr)
    missing(gtk);
  const TrackInformation *tkInfo = dynamic_cast<const TrackInformation *>(gui);
  if (tkInfo == nullptr)
    wrongType();
  return *tkInfo;
}

TrackInformation &TrackInformationExtractor::operator()(G4Track &gtk) const {
  G4VUserTrackInformation *gui = gtk.GetUserInformation();
  if (gui == nullptr)
    missing(gtk);
  TrackInformation *tkInfo = dynamic_cast<TrackInformation *>(gui);
  if (tkInfo == nullptr)
    wrongType();
  return *tkInfo;
}
