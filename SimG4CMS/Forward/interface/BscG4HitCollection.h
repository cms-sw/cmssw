///////////////////////////////////////////////////////////////////////////////
// File: BscG4HitCollection.h
// Date: 02.2006
// Description: Bsc detector Hit collection
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#ifndef BscG4HitCollection_h
#define BscG4HitCollection_h

#include "G4THitsCollection.hh"
#include "SimG4CMS/Forward/interface/BscG4Hit.h"
#include "G4Step.hh"
#include <boost/cstdint.hpp>

typedef G4THitsCollection<BscG4Hit> BscG4HitCollection;

#endif

