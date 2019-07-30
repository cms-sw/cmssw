#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/DDG4Builder.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "SimG4Core/Geometry/interface/DDG4ProductionCuts.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "SimG4Core/Geometry/interface/DD4hep_DDG4Builder.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DDG4/Geant4Converter.h"
#include "DDG4/Geant4GeometryInfo.h"
#include "DDG4/Geant4Mapping.h"
#include "DD4hep/Detector.h"
#include "DD4hep/Printout.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4PVPlacement.hh"

using namespace edm;
using namespace dd4hep;
using namespace dd4hep::sim;

DDDWorld::DDDWorld(const DDCompactView *pDD,
                   const cms::DDCompactView *pDD4hep,
                   SensitiveDetectorCatalog &catalog,
                   int verb,
                   bool cuts,
                   bool pcut) {
  LogVerbatim("SimG4CoreApplication") << "+++ DDDWorld: initialisation of Geant4 geometry";
  if (pDD4hep) {
    // DD4Hep
    const cms::DDDetector *det = pDD4hep->detector();
    dd4hep::sim::Geant4GeometryMaps::VolumeMap lvMap;

    cms::DDG4Builder theBuilder(pDD4hep, lvMap, false);
    m_world = theBuilder.BuildGeometry(catalog);
    LogVerbatim("SimG4CoreApplication") << "DDDWorld: worldLV: " << m_world->GetName();
    if (cuts) {
      DDG4ProductionCuts pcuts(&det->specpars(), &lvMap, verb, pcut);
    }
    catalog.printMe();
  } else {
    // old DD code
    G4LogicalVolumeToDDLogicalPartMap lvMap;

    DDG4Builder theBuilder(pDD, lvMap, false);
    G4LogicalVolume *world = theBuilder.BuildGeometry(catalog);
    LogVerbatim("SimG4CoreApplication") << "DDDWorld: worldLV: " << world->GetName();
    m_world = new G4PVPlacement(nullptr, G4ThreeVector(), world, "DDDWorld", nullptr, false, 0);
    if (cuts) {
      DDG4ProductionCuts pcuts(&lvMap, verb, pcut);
    }
  }
  LogVerbatim("SimG4CoreApplication") << "DDDWorld: initialisation of Geant4 geometry is done.";
}

DDDWorld::DDDWorld(const DDCompactView *cpv,
                   G4LogicalVolumeToDDLogicalPartMap &lvmap,
                   SensitiveDetectorCatalog &catalog,
                   bool check) {
  LogVerbatim("SimG4CoreApplication") << "DDDWorld: initialization of Geant4 geometry";
  DDG4Builder theBuilder(cpv, lvmap, check);

  G4LogicalVolume *world = theBuilder.BuildGeometry(catalog);

  m_world = new G4PVPlacement(nullptr, G4ThreeVector(), world, "DDDWorld", nullptr, false, 0);
  LogVerbatim("SimG4CoreApplication") << "DDDWorld: initialization of Geant4 geometry is done.";
}

DDDWorld::~DDDWorld() {}
