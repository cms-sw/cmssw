///////////////////////////////////////////////////////////////////////////////
// File: FP420G4HitCollection.h
// Date: 02.2006
// Description: FP420 detector Hit collection
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#ifndef FP420G4HitCollection_h
#define FP420G4HitCollection_h

#include "G4THitsCollection.hh"
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "G4Step.hh"

typedef G4THitsCollection<FP420G4Hit> FP420G4HitCollection;

#endif
