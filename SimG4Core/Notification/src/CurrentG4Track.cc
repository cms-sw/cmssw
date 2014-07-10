#include "SimG4Core/Notification/interface/CurrentG4Track.h"

thread_local const G4Track * CurrentG4Track::m_track = 0;
thread_local bool CurrentG4Track::m_tracking = false;

void CurrentG4Track::setTrack(const G4Track * t)
{
    if (m_tracking) 
	throw SimG4Exception("CurrentG4Track: new track set while previous is being tracked");
    m_track = t;
    m_tracking = true;
}

void CurrentG4Track::postTracking(const G4Track * t) 
{
    if (t != m_track)
	throw SimG4Exception("CurrentG4Track: tracking finishes for a different track");
    if (!m_tracking)
	throw SimG4Exception("CurrentG4Track: tracking finishes without having started");
    m_tracking = false;
}

