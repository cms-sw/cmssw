#ifndef _PPS_PixelG4HitCollection_h
#define _PPS_PixelG4HitCollection_h 1
// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelG4HitCollection
//
/**\class PPSPixelG4HitCollection PPSPixelG4HitCollection.h SimG4CMS/PPS/interface/PPSPixelG4HitCollection.h
 
 Description: Hit collection class for PPS transient hits
 
 Usage: 
 
*/
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files
#include "SimG4CMS/PPS/interface/PPSPixelG4Hit.h"
#include "G4THitsCollection.hh"

typedef G4THitsCollection<PPSPixelG4Hit> PPSPixelG4HitCollection;

#endif
