#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

namespace simcluster_utils {
  const std::unordered_map<std::string, std::vector<DetId::Detector>> DetIdMap = {
      {"", std::vector<DetId::Detector>{}},
      {"Ecal", std::vector<DetId::Detector>{DetId::Ecal}},
      {"Hcal", std::vector<DetId::Detector>{DetId::Hcal}},
      {"HGCal", std::vector<DetId::Detector>{DetId::HGCalEE, DetId::HGCalHSi, DetId::HGCalHSc}}};

  // Build the merged set of detectors from all requested filter strings
  std::vector<DetId::Detector> check_and_join_detids(const std::vector<std::string>& dets_v) {
    std::vector<DetId::Detector> detIds;
    for (const auto& det : dets_v) {
      if (simcluster_utils::DetIdMap.find(det) == simcluster_utils::DetIdMap.end()) {
        throw cms::Exception("Configuration") << "dets_v: unknown value '" << det << "'. "
                                              << "Allowed values are: Ecal, Hcal, HGCal, or empty string.";
      }
      const auto& dets = simcluster_utils::DetIdMap.at(det);
      detIds.insert(detIds.end(), dets.begin(), dets.end());
    }
    return detIds;
  }
}  // namespace simcluster_utils
