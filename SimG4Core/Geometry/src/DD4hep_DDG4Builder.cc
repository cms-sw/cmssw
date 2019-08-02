#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Geometry/interface/DD4hep_DDG4Builder.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/Filter.h"
#include "DDG4/Geant4Converter.h"
#include "DDG4/Geant4GeometryInfo.h"
#include "DDG4/Geant4Mapping.h"
#include "DD4hep/Detector.h"

#include "G4LogicalVolume.hh"
#include "G4ReflectionFactory.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  Geant4Converter g4Geo(detector);
  Geant4GeometryInfo *geometry = g4Geo.create(world).detach();
  map_ = geometry->g4Volumes;

  std::vector<std::pair<G4LogicalVolume *, const cms::DDSpecPar *>> dd4hepVec;
  const cms::DDSpecParRegistry &specPars = det->specpars();
  cms::DDSpecParRefs specs;
  specPars.filter(specs, "SensitiveDetector");
  for (auto const &it : map_) {
    for (auto const &fit : specs) {
      for (auto const &pit : fit->paths) {
        if (cms::dd::compareEqual(cms::dd::noNamespace(it.first.name()), cms::dd::realTopName(pit))) {
          dd4hepVec.emplace_back(&*it.second, &*fit);
        }
      }
    }
  }

  for (auto const &it : dd4hepVec) {
    auto sClassName = it.second->strValue("SensitiveDetector");
    auto sROUName = it.second->strValue("ReadOutName");
    auto fff = it.first->GetName();
    catalog.insert({sClassName.data(), sClassName.size()}, {sROUName.data(), sROUName.size()}, fff);

    edm::LogVerbatim("SimG4CoreApplication")
        << " DDG4SensitiveConverter: Sensitive " << fff << " Class Name " << sClassName << " ROU Name " << sROUName;
  }

  return geometry->world();
}
