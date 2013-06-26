#include "SimG4Core/Geometry/interface/DDG4SensitiveConverter.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4LogicalVolume.hh"

using std::string;
using std::vector;
using std::cout;
using std::endl;

DDG4SensitiveConverter::DDG4SensitiveConverter() {}

DDG4SensitiveConverter::~DDG4SensitiveConverter() {}

SensitiveDetectorCatalog DDG4SensitiveConverter::upDate(const DDG4DispContainer & ddg4s) {

  LogDebug("SimG4CoreGeometry") <<" DDG4SensitiveConverter::upDate() starts" ;
  SensitiveDetectorCatalog catalog;

  for (unsigned int i=0; i<ddg4s.size(); i++)  {
    DDG4Dispatchable * ddg4 = ddg4s[i];
    const DDLogicalPart * part   = (ddg4->getDDLogicalPart());
    G4LogicalVolume *     result = (ddg4->getG4LogicalVolume());
  
    std::string sClassName = getString("SensitiveDetector",part);
    std::string sROUName   = getString("ReadOutName",part);
    std::string fff        = result->GetName();
    if (sClassName != "NotFound") {
      LogDebug("SimG4CoreGeometry") << " DDG4SensitiveConverter: Sensitive " << fff
				    << " Class Name " << sClassName << " ROU Name " << sROUName ;	    
      fff = result->GetName();
      catalog.insert(sClassName,sROUName,fff);
    }
  }
  return catalog;
}

std::string DDG4SensitiveConverter::getString(const std::string & s, 
					      const DDLogicalPart * part) {
  std::vector<std::string> temp;
  DDValue val(s);
  std::vector<const DDsvalues_type *> result = part->specifics();
  std::vector<const DDsvalues_type *>::iterator it = result.begin();
  bool foundIt = false;
  for (; it != result.end(); ++it) {
    foundIt = DDfetch(*it,val);
    if (foundIt) break;
  }    
  if (foundIt) { 
    temp = val.strings(); 
    if (temp.size() != 1) {
      edm::LogError("SimG4CoreGeometry") << "DDG4SensitiveConverter - ERROR: I need 1 " << s << " tags" ;
      throw SimG4Exception("DDG4SensitiveConverter: Problem with Region tags: one and only one is allowed");
    }
    return temp[0]; 
  }
  return "NotFound";
}

