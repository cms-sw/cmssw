#include "SimG4Core/Notification/interface/CurrentG4Track.h"

int CurrentG4Track::m_nWorkers = 0;
thread_local const G4Track* CurrentG4Track::m_track = nullptr;

void CurrentG4Track::setTrack(const G4Track* t) { m_track = t; }

const G4Track* CurrentG4Track::track() { return m_track; }

int CurrentG4Track::numberOfWorkers() { return m_nWorkers; };

void CurrentG4Track::setNumberOfWorkers(int n) { m_nWorkers = n; };
