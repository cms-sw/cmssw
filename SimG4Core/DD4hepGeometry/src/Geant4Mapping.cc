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
//FIXME: #include "DDG4/Geant4Mapping.h"
#include "SimG4Core/DD4hepGeometry/interface/Geant4Mapping.h"

#include "DD4hep/Printout.h"
#include "DD4hep/VolumeManager.h"
#include "G4PVPlacement.hh"
#include <stdexcept>

using namespace dd4hep::sim;
using namespace dd4hep;
using namespace std;

/// Initializing Constructor
Geant4Mapping::Geant4Mapping(const Detector& description_ref)
  : m_detDesc(description_ref), m_dataPtr(0) {
}

/// Standard destructor
Geant4Mapping::~Geant4Mapping() {
  if (m_dataPtr)
    delete m_dataPtr;
  m_dataPtr = 0;
}

/// Possibility to define a singleton instance
Geant4Mapping& Geant4Mapping::instance() {
  static Geant4Mapping inst(Detector::getInstance());
  return inst;
}

/// When resolving pointers, we must check for the validity of the data block
void Geant4Mapping::checkValidity() const {
  if (m_dataPtr)
    return;
  throw runtime_error("Geant4Mapping: Attempt to access an invalid data block!");
}

/// Create new data block. Delete old data block if present.
Geant4GeometryInfo& Geant4Mapping::init() {
  Geant4GeometryInfo* p = detach();
  if (p)
    delete p;
  attach(new Geant4GeometryInfo());
  return data();
}

/// Release data and pass over the ownership
Geant4GeometryInfo* Geant4Mapping::detach() {
  Geant4GeometryInfo* p = m_dataPtr;
  m_dataPtr = 0;
  return p;
}

/// Set a new data block
void Geant4Mapping::attach(Geant4GeometryInfo* data_ptr) {
  m_dataPtr = data_ptr;
}

/// Access the volume manager
//FIXME: Geant4VolumeManager Geant4Mapping::volumeManager() const {
//   if ( m_dataPtr ) {
//     if ( m_dataPtr->g4Paths.empty() ) {
//       VolumeManager::getVolumeManager(m_detDesc);
//       return Geant4VolumeManager(m_detDesc, m_dataPtr);
//     }
//     return Geant4VolumeManager(Handle < Geant4GeometryInfo > (m_dataPtr));
//   }
//   throw runtime_error(format("Geant4Mapping", "Cannot create volume manager without Geant4 geometry info [Invalid-Info]"));
// }

/// Accessor to resolve geometry placements
PlacedVolume Geant4Mapping::placement(const G4VPhysicalVolume* node) const {
  checkValidity();
  const Geant4GeometryMaps::PlacementMap& m = m_dataPtr->g4Placements;
  for (Geant4GeometryMaps::PlacementMap::const_iterator i = m.begin(); i != m.end(); ++i)
    if ((*i).second == node)
      return PlacedVolume((*i).first);
  return PlacedVolume(0);
}
