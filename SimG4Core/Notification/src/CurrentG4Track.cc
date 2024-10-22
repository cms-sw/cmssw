#include "SimG4Core/Notification/interface/CurrentG4Track.h"

thread_local const G4Track* CurrentG4Track::m_track = nullptr;

void CurrentG4Track::setTrack(const G4Track* t) { m_track = t; }

const G4Track* CurrentG4Track::track() { return m_track; }
