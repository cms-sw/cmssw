#ifndef SimG4CMS_CaloG4HitCollection_h
#define SimG4CMS_CaloG4HitCollection_h 1
///////////////////////////////////////////////////////////////////////////////
// File: CaloG4HitCollection.h
// Description: Calorimeter Hit collection
///////////////////////////////////////////////////////////////////////////////

#include "G4THitsCollection.hh"
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"

typedef G4THitsCollection<CaloG4Hit> CaloG4HitCollection;

#endif
