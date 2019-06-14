#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#define DEBUG

#include <iostream>

void SensitiveDetectorCatalog::insert(const std::string &cN, const std::string &rN, const std::string &lvN) {
  theClassNameMap[cN].emplace_back(rN);
  theROUNameMap[rN].emplace_back(lvN);
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
  for (auto const& it : theROUNameMap)
    temp.emplace_back(it.first);
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
  for (auto const& it : rous)
    temp.emplace_back(it);
  return temp;
}

std::string SensitiveDetectorCatalog::className(const std::string &readoutName) const {
  for (auto const& it : theClassNameMap) {
    std::vector<std::string> temp = it.second;
    for (auto const& it2 : temp) {
      if (it2 == readoutName)
        return it.first;
    }
  }
  return "NotFound";
}

std::vector<std::string> SensitiveDetectorCatalog::classNames() const {
  std::vector<std::string> temp;
  for (auto const& it : theClassNameMap)
    temp.emplace_back(it.first);
  return temp;
}
