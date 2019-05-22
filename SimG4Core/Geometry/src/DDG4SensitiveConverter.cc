#include "SimG4Core/Geometry/interface/DDG4SensitiveConverter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4LogicalVolume.hh"

DDG4SensitiveConverter::DDG4SensitiveConverter() {}

DDG4SensitiveConverter::~DDG4SensitiveConverter() {}

SensitiveDetectorCatalog DDG4SensitiveConverter::upDate(const DDG4DispContainer &ddg4s) {
  LogDebug("SimG4CoreGeometry") << " DDG4SensitiveConverter::upDate() starts";
  SensitiveDetectorCatalog catalog;

  for (auto ddg4 : ddg4s) {
    const DDLogicalPart *part = (ddg4->getDDLogicalPart());
    G4LogicalVolume *result = (ddg4->getG4LogicalVolume());

    std::string sClassName = getString("SensitiveDetector", part);
    std::string sROUName = getString("ReadOutName", part);
    std::string fff = result->GetName();
    if (sClassName != "NotFound") {
      LogDebug("SimG4CoreGeometry") << " DDG4SensitiveConverter: Sensitive " << fff << " Class Name " << sClassName
                                    << " ROU Name " << sROUName;
      fff = result->GetName();
      catalog.insert(sClassName, sROUName, fff);
    }
  }
  return catalog;
}

std::string DDG4SensitiveConverter::getString(const std::string &ss, const DDLogicalPart *part) {
  std::vector<std::string> temp;
  DDValue val(ss);
  std::vector<const DDsvalues_type *> result = part->specifics();
  bool foundIt = false;
  for (auto stype : result) {
    foundIt = DDfetch(stype, val);
    if (foundIt)
      break;
  }
  if (foundIt) {
    temp = val.strings();
    if (temp.size() != 1) {
      edm::LogError("SimG4CoreGeometry") << "DDG4SensitiveConverter - ERROR: I need 1 " << ss << " tags";
      throw cms::Exception("SimG4CoreGeometry",
                           " DDG4SensitiveConverter::getString Problem with "
                           "Region tags - one and only one allowed: " +
                               ss);
    }
    return temp[0];
  }
  return "NotFound";
}
