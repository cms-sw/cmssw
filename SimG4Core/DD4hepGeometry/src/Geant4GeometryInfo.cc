//==========================================================================
//  AIDA Detector description implementation 
//--------------------------------------------------------------------------
// Copyright (C) Organisation europeenne pour la Recherche nucleaire (CERN)
// All rights reserved.
//
// For the licensing terms see $DD4hepINSTALL/LICENSE.
// For the list of contributors see $DD4hepINSTALL/doc/CREDITS.
//
// Author     : M.Frank
//
//==========================================================================

// Framework include files
//FIXME: #include "DDG4/Geant4GeometryInfo.h"
#include "SimG4Core/DD4hepGeometry/interface/Geant4GeometryInfo.h"

// Geant4 include files
#include "G4VPhysicalVolume.hh"

// C/C++ include files
#include <stdexcept>

using namespace std;
using namespace dd4hep::sim;


string Geant4GeometryInfo::placementPath(const Geant4PlacementPath& path, bool reverse)   {
  string s;
  if ( reverse )  {
    for (Geant4PlacementPath::const_reverse_iterator pIt = path.rbegin(); pIt != path.rend(); ++pIt) {
      s += "/"; s += (*pIt)->GetName();
    }
  }
  else  {
    for (Geant4PlacementPath::const_iterator pIt = path.begin(); pIt != path.end(); ++pIt) {
      s += "/"; s += (*pIt)->GetName();
    }
  }
  return s;
}

/// Default constructor
Geant4GeometryInfo::Geant4GeometryInfo()
  : TNamed("Geant4GeometryInfo", "Geant4GeometryInfo"), m_world(0), printLevel(DEBUG), valid(false) {
}

/// Default destructor
Geant4GeometryInfo::~Geant4GeometryInfo() {
}

/// The world placement
G4VPhysicalVolume* Geant4GeometryInfo::world() const   {
  if ( m_world ) return m_world;
  throw runtime_error("Geant4GeometryInfo: Attempt to access invalid world placement");
}

/// Set the world placement
void Geant4GeometryInfo::setWorld(const TGeoNode* node)    {
  Geant4GeometryMaps::PlacementMap::const_iterator g4it = g4Placements.find(node);
  G4VPhysicalVolume* g4 = (g4it == g4Placements.end()) ? 0 : (*g4it).second;
  if (!g4) {
    throw runtime_error("Geant4GeometryInfo: Attempt to SET invalid world placement");
  }
  m_world = g4;
}
