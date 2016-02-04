#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define DEBUG

#include <iostream>

void SensitiveDetectorCatalog::insert(std::string& cN, std::string& rN, 
				       std::string& lvN) {
  theClassNameMap[cN].push_back(rN);
  theROUNameMap[rN].push_back(lvN);
#ifdef DEBUG
  LogDebug("SimG4CoreGeometry") << "SenstiveDetectorCatalog: insert (" << cN
				<< "," << rN << "," << lvN << ")\n"
				<< "                         has     "
				<< readoutNames().size() << " ROUs "
				<< readoutNames().front() << "\n"
				<< "                         has     "
				<< classNames().size() << " classes "
				<< classNames().front();
#endif
}

std::vector<std::string> SensitiveDetectorCatalog::readoutNames() {
  std::vector<std::string> temp;
  for (MapType::const_iterator it = theROUNameMap.begin();
       it != theROUNameMap.end(); it++)
    temp.push_back(it->first);
  return temp;
}

std::vector<std::string> SensitiveDetectorCatalog::readoutNames(std::string & className) { 
  return theClassNameMap[className]; 
}

std::vector<std::string> SensitiveDetectorCatalog::logicalNames(std::string & readoutName) { 
  return theROUNameMap[readoutName];
}

std::vector<std::string> SensitiveDetectorCatalog::logicalNamesFromClassName(std::string & className) {
  std::vector<std::string> temp;
  std::vector<std::string> rous = theClassNameMap[className];
  for (std::vector<std::string>::const_iterator it = rous.begin(); 
       it!= rous.end(); it++)
    temp.push_back(*it);
  return temp;
}

std::string SensitiveDetectorCatalog::className(std::string & readoutName) {
  for (MapType::const_iterator it = theClassNameMap.begin();  
       it != theClassNameMap.end(); it++)  {
    std::vector<std::string> temp = (*it).second;
    for (std::vector<std::string>::const_iterator it2 = temp.begin();
	 it2!=temp.end(); it2++) {
      if (*it2 == readoutName ) return (*it).first;
    }
  } 
  return "NotFound";
}

std::vector<std::string> SensitiveDetectorCatalog::classNames() {
  std::vector<std::string> temp;
  for (MapType::const_iterator it = theClassNameMap.begin();  
       it != theClassNameMap.end(); it++)
    temp.push_back(it->first);
  return temp;
}
