#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

namespace simcluster_utils {
  const std::unordered_map<std::string, std::vector<DetId::Detector>> DetIdMap = {
      {"", std::vector<DetId::Detector>{}},
      {"Ecal", std::vector<DetId::Detector>{DetId::Ecal}},
      {"Hcal", std::vector<DetId::Detector>{DetId::Hcal}},
      {"HGCal", std::vector<DetId::Detector>{DetId::HGCalEE, DetId::HGCalHSi, DetId::HGCalHSc}}};

  // Build the merged set of detectors from all requested filter strings
  std::vector<DetId::Detector> join_detids(const std::vector<std::string>& dets_v) {
    std::vector<DetId::Detector> detIds;
    for (const auto& det : dets_v) {
      const auto& dets = simcluster_utils::DetIdMap.at(det);
      detIds.insert(detIds.end(), dets.begin(), dets.end());
    }
    return detIds;
  }

  void check_detids(const std::vector<std::string>& dets_v) {
    std::vector<std::string> allowed(simcluster_utils::DetIdMap.size());
    int i = 0;
    for (auto it = simcluster_utils::DetIdMap.begin(); it != simcluster_utils::DetIdMap.end(); ++it) {
      allowed[i] = it->first;  // store key
      ++i;
    }

    for (const auto& det : dets_v) {
      if (std::find(allowed.begin(), allowed.end(), det) == allowed.end()) {
        throw cms::Exception("Configuration") << "dets_v: unknown value '" << det << "'. "
                                              << "Allowed values are: Ecal, Hcal, HGCal, or empty string.";
      }
    }
  }
}  // namespace simcluster_utils
