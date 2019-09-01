#ifndef Forward_TotemG4HitCollection_h
#define Forward_TotemG4HitCollection_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemG4HitCollection
//
/**\class TotemG4HitCollection TotemG4HitCollection.h SimG4CMS/Forward/interface/TotemG4HitCollection.h
 
 Description: Hit collection class for Totem transient hits
 
 Usage: 
 
*/
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files
#include "SimG4CMS/Forward/interface/TotemG4Hit.h"
#include "G4THitsCollection.hh"

typedef G4THitsCollection<TotemG4Hit> TotemG4HitCollection;

#endif
