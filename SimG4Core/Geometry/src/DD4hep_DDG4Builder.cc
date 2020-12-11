#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Geometry/interface/DD4hep_DDG4Builder.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include <DDG4/Geant4Converter.h>
#include <DDG4/Geant4GeometryInfo.h>
#include <DDG4/Geant4Mapping.h>
#include <DD4hep/Detector.h>
#include <DD4hep/Filter.h>

#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4ReflectionFactory.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

using namespace cms;
using namespace dd4hep;
using namespace dd4hep::sim;

DDG4Builder::DDG4Builder(const cms::DDCompactView *cpv, dd4hep::sim::Geant4GeometryMaps::VolumeMap &lvmap, bool check)
    : compactView_(cpv), map_(lvmap), check_(check) {}

G4VPhysicalVolume *DDG4Builder::BuildGeometry(SensitiveDetectorCatalog &catalog) {
  G4ReflectionFactory *refFact = G4ReflectionFactory::Instance();
  refFact->SetScalePrecision(100. * refFact->GetScalePrecision());

  const cms::DDDetector *det = compactView_->detector();
  DetElement world = det->description()->world();
  const Detector &detector = *det->description();

  // GET FULL G4 GEOMETRY
  Geant4Converter g4Geo(detector);
  g4Geo.debugMaterials = false;
  Geant4GeometryInfo *geometry = g4Geo.create(world).detach();
  map_ = geometry->g4Volumes;

  // FIND & STORE ALL G4 LOGICAL VOLUMES DEFINED AS SENSITIVE IN CMSSW XMLS
  std::vector<std::pair<G4LogicalVolume *, const dd4hep::SpecPar *>> dd4hepVec;
  const dd4hep::SpecParRegistry &specPars = det->specpars();
  dd4hep::SpecParRefs specs;
  specPars.filter(specs, "SensitiveDetector");
  for (auto const &it : map_) {
    for (auto const &fit : specs) {
      for (auto const &pit : fit.second->paths) {
        if (dd4hep::dd::compareEqualName(dd4hep::dd::realTopName(pit), dd4hep::dd::noNamespace(it.first.name()))) {
          dd4hepVec.emplace_back(&*it.second, &*fit.second);
        }
      }
    }
  }

  // ADD ALL SELECTED G4 LOGICAL VOLUMES TO SENSITIVE DETECTORS CATALOGUE
  for (auto const &it : dd4hepVec) {
    // Sensitive detector info
    const G4String &sensitiveDetectorG4Name = it.first->GetName();
    auto sClassName = it.second->strValue("SensitiveDetector");
    auto sROUName = it.second->strValue("ReadOutName");
    // Add to catalogue
    catalog.insert({sClassName.data(), sClassName.size()}, {sROUName.data(), sROUName.size()}, sensitiveDetectorG4Name);

    edm::LogVerbatim("SimG4CoreApplication") << " DDG4SensitiveConverter: Sensitive " << sensitiveDetectorG4Name
                                             << " Class Name " << sClassName << " ROU Name " << sROUName;

    // Reflected sensors also need to be added to the senstive detectors catalogue!
    // Similar treatment here with DD4hep, as what was done for old DD.
    const G4String &sensitiveDetectorG4ReflectedName = sensitiveDetectorG4Name + "_refl";

    const G4LogicalVolumeStore *const allG4LogicalVolumes = G4LogicalVolumeStore::GetInstance();
    const bool hasG4ReflectedVolume =
        std::find_if(
            allG4LogicalVolumes->begin(), allG4LogicalVolumes->end(), [&](G4LogicalVolume *const aG4LogicalVolume) {
              return (aG4LogicalVolume->GetName() == sensitiveDetectorG4ReflectedName);
            }) != allG4LogicalVolumes->end();
    if (hasG4ReflectedVolume) {
      // Add reflected sensitive detector to catalogue
      catalog.insert(
          {sClassName.data(), sClassName.size()}, {sROUName.data(), sROUName.size()}, sensitiveDetectorG4ReflectedName);

      edm::LogVerbatim("SimG4CoreApplication")
          << " DDG4SensitiveConverter: Sensitive " << sensitiveDetectorG4ReflectedName << " Class Name " << sClassName
          << " ROU Name " << sROUName;
    }
  }

  return geometry->world();
}
