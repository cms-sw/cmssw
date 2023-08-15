#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#define EDM_ML_DEBUG

#include <iostream>

void SensitiveDetectorCatalog::insert(const std::string &cN, const std::string &rN, const std::string &lvN) {
  theClassNameMap[cN].insert(rN);
  theROUNameMap[rN].insert(lvN);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("SensitiveDetector") << "SenstiveDetectorCatalog: insert (" << cN << "," << rN << "," << lvN << ")\n"
                                        << "                         has     " << readoutNames().size() << " ROUs "
                                        << readoutNames().front() << "\n"
                                        << "                         has     " << classNames().size() << " classes "
                                        << classNames().front();
#endif
}

std::vector<std::string_view> SensitiveDetectorCatalog::readoutNames() const {
  std::vector<std::string_view> temp;
  for (auto const &it : theROUNameMap)
    temp.emplace_back(it.first);
  return temp;
}

const std::vector<std::string_view> SensitiveDetectorCatalog::readoutNames(const std::string &className) const {
  return std::vector<std::string_view>(theClassNameMap.at(className).begin(), theClassNameMap.at(className).end());
}

const std::vector<std::string_view> SensitiveDetectorCatalog::logicalNames(const std::string &readoutName) const {
  return std::vector<std::string_view>(theROUNameMap.at(readoutName).begin(), theROUNameMap.at(readoutName).end());
}

std::vector<std::string_view> SensitiveDetectorCatalog::logicalNamesFromClassName(const std::string &className) const {
  std::vector<std::string_view> temp;
  const std::vector<std::string_view> rous(theClassNameMap.at(className).begin(), theClassNameMap.at(className).end());
  temp.reserve(rous.size());
  for (auto const &it : rous)
    temp.emplace_back(it);
  return temp;
}

std::string_view SensitiveDetectorCatalog::className(const std::string &readoutName) const {
  for (auto const &it : theClassNameMap) {
    std::vector<std::string_view> temp(it.second.begin(), it.second.end());
    for (auto const &it2 : temp) {
      if (it2 == readoutName)
        return it.first;
    }
  }
  return "NotFound";
}

std::vector<std::string_view> SensitiveDetectorCatalog::classNames() const {
  std::vector<std::string_view> temp;
  for (auto const &it : theClassNameMap)
    temp.emplace_back(it.first);
  return temp;
}

void SensitiveDetectorCatalog::printMe() const {
  edm::LogVerbatim("SensitiveDetector") << "Class names map size is: " << theClassNameMap.size() << "\n";
  edm::LogVerbatim("SensitiveDetector").log([&](auto &log) {
    int i(0);
    for (const auto &cn : theClassNameMap) {
      log << "#" << ++i << ": " << cn.first << " has " << cn.second.size() << " class names:\n";
      for (const auto &cnv : cn.second)
        log << cnv << ", ";
      log << "\n";
    }
    log << "\n";
  });
  edm::LogVerbatim("SensitiveDetector") << "\nROU names map: " << theROUNameMap.size() << "\n";
  edm::LogVerbatim("SensitiveDetector").log([&](auto &log) {
    int i(0);
    for (const auto &rn : theROUNameMap) {
      log << "#" << ++i << ": " << rn.first << " has " << rn.second.size() << " ROU names:\n";
      for (const auto &rnv : rn.second)
        log << rnv << "\n ";
      log << "\n";
    }
    log << "\n";
  });

  edm::LogVerbatim("SensitiveDetector") << "\n========== Here are the accessors =================\n";
  edm::LogVerbatim("SensitiveDetector").log([&](auto &log) {
    for (auto c : classNames()) {
      log << "ClassName:" << c << "\n";
      for (auto r : readoutNames({c.data(), c.size()})) {
        log << "       RedoutName:" << r << "\n";
        for (auto l : logicalNames({r.data(), r.size()})) {
          log << "              LogicalName:" << l << "\n";
        }
      }
    }
  });
}
