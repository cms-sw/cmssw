#include "DataFormats/DetId/interface/DetId.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloValidationStatistics.h"

#include <iostream>

CaloHitAnalyzer::CaloHitAnalyzer(const std::string &name,
                                 double hitEnergyThreshold,
                                 const CaloVSimParameterMap *parameterMap,
                                 const CaloVHitFilter *filter)
    : hitEnergySumMap_(),
      hitEnergyThreshold_(hitEnergyThreshold),
      simParameterMap_(parameterMap),
      hitFilter_(filter),
      summary_(name, 1., 0.),
      noiseHits_(0) {}

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
void CaloHitAnalyzer::fillHits(MixCollection<PCaloHit> &hits) {
  hitEnergySumMap_.clear();
  noiseHits_ = 0;
  // store the energy of each hit in a map
  MixCollection<PCaloHit>::MixItr hitItr = hits.begin();
  MixCollection<PCaloHit>::MixItr last = hits.end();
  for (; hitItr != last; ++hitItr) {
    if (hitFilter_ == nullptr || hitFilter_->accepts(*hitItr)) {
      int id = hitItr->id();
      // double samplingFactor =
      // simParameterMap_->simParameters(DetId(id)).samplingFactor();
      double samplingFactor = 1.;
      double energy = hitItr->energy() * samplingFactor;

      // add it to the map
      std::map<int, double>::iterator mapItr = hitEnergySumMap_.find(id);
      if (mapItr == hitEnergySumMap_.end()) {
        hitEnergySumMap_[id] = energy;
      } else {
        mapItr->second += energy;
      }
    }
  }
}

void CaloHitAnalyzer::analyze(int id, double recEnergy) {
  if (recEnergy > hitEnergyThreshold_) {
    std::map<int, double>::iterator mapItr = hitEnergySumMap_.find(id);
    if (mapItr == hitEnergySumMap_.end()) {
      ++noiseHits_;
    } else {
      // keep statistics of the rec energy / sim energy
      summary_.addEntry(recEnergy / mapItr->second);
    }
  }
}
