#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#define DEBUG

#include <iostream>

void SensitiveDetectorCatalog::insert(const std::string &cN, const std::string &rN, const std::string &lvN) {
  theClassNameMap[cN].push_back(rN);
  theROUNameMap[rN].push_back(lvN);
#ifdef DEBUG
  LogDebug("SimG4CoreGeometry") << "SenstiveDetectorCatalog: insert (" << cN << "," << rN << "," << lvN << ")\n"
                                << "                         has     " << readoutNames().size() << " ROUs "
                                << readoutNames().front() << "\n"
                                << "                         has     " << classNames().size() << " classes "
                                << classNames().front();
#endif
}

std::vector<std::string> SensitiveDetectorCatalog::readoutNames() const {
  std::vector<std::string> temp;
  for (MapType::const_iterator it = theROUNameMap.begin(); it != theROUNameMap.end(); it++)
    temp.push_back(it->first);
  return temp;
}

const std::vector<std::string> &SensitiveDetectorCatalog::readoutNames(const std::string &className) const {
  return theClassNameMap.at(className);
}

const std::vector<std::string> &SensitiveDetectorCatalog::logicalNames(const std::string &readoutName) const {
  return theROUNameMap.at(readoutName);
}

std::vector<std::string> SensitiveDetectorCatalog::logicalNamesFromClassName(const std::string &className) const {
  std::vector<std::string> temp;
  const std::vector<std::string> &rous = theClassNameMap.at(className);
  for (std::vector<std::string>::const_iterator it = rous.begin(); it != rous.end(); it++)
    temp.push_back(*it);
  return temp;
}

std::string SensitiveDetectorCatalog::className(const std::string &readoutName) const {
  for (MapType::const_iterator it = theClassNameMap.begin(); it != theClassNameMap.end(); it++) {
    std::vector<std::string> temp = (*it).second;
    for (std::vector<std::string>::const_iterator it2 = temp.begin(); it2 != temp.end(); it2++) {
      if (*it2 == readoutName)
        return (*it).first;
    }
  }
  return "NotFound";
}

std::vector<std::string> SensitiveDetectorCatalog::classNames() const {
  std::vector<std::string> temp;
  for (MapType::const_iterator it = theClassNameMap.begin(); it != theClassNameMap.end(); it++)
    temp.push_back(it->first);
  return temp;
}
