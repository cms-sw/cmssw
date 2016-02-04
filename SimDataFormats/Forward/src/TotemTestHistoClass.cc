// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemTestHistoClass
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemTestHistoClass.cc,v 1.1 2006/11/16 16:40:32 sunanda Exp $
//

// system include files
#include <iostream>
#include <cmath>

// user include files
#include "SimDataFormats/Forward/interface/TotemTestHistoClass.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//

TotemTestHistoClass::TotemTestHistoClass() : evt(0), hits(0) {}

TotemTestHistoClass::~TotemTestHistoClass() {}

void TotemTestHistoClass::fillHit(int uID, int pType, int tID, int pID, 
				  float eLoss, float pAbs, float vX, float vY,
				  float vZ, float x, float y, float z) {

  TotemTestHistoClass::Hit h;
  h.UID   = uID;
  h.Ptype = pType;
  h.TID   = tID;
  h.PID   = pID;
  h.ELoss = eLoss;
  h.PABS  = pAbs;
  h.x     = x;
  h.y     = y;
  h.z     = z;
  h.vx    = vX;
  h.vy    = vY;
  h.vz    = vZ;
  hit.push_back(h);
  hits++;
  LogDebug("ForwardSim") << "TotemTestHistoClass : Hit " << hits << " " << uID
			 << ", " << pType << ", " << tID << ", " << pID << ", "
			 << eLoss << ", " << pAbs << ", " << vX << ", " << vY
			 << ", " << vZ << ", " << x << ", " << y << ", " << z;
}
