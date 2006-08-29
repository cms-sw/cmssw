#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloValidationStatistics.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <iostream>
using std::map;
using std::string;
using std::cout;
using std::endl;
using edm::PCaloHitContainer;


CaloHitAnalyzer::CaloHitAnalyzer(const string & name,
                      double hitEnergyThreshold,
                      const CaloVSimParameterMap * parameterMap,
                      const CaloVHitFilter * filter)
: hitEnergySumMap_(),
  hitEnergyThreshold_(hitEnergyThreshold),
  simParameterMap_(parameterMap),
  hitFilter_(filter),
  summary_(name, 1., 0.),
  noiseHits_(0)
{
}


#include "DataFormats/HcalDetId/interface/HcalDetId.h"
void CaloHitAnalyzer::fillHits(const PCaloHitContainer & hits) {
  hitEnergySumMap_.clear();
  noiseHits_ = 0;
  // store the energy of each hit in a map
  PCaloHitContainer::const_iterator hitItr = hits.begin();
  PCaloHitContainer::const_iterator last = hits.end();
  for( ; hitItr != last; ++hitItr) 
  {
    if(hitFilter_ == 0 || hitFilter_->accepts(*hitItr)) {
      int id = hitItr->id();
      double samplingFactor = simParameterMap_->simParameters(DetId(id)).samplingFactor();
      double energy = hitItr->energy() * samplingFactor;

      // add it to the map
      map<int, double>::iterator mapItr = hitEnergySumMap_.find(id);
      if(mapItr == hitEnergySumMap_.end()) {
        hitEnergySumMap_[id] = energy;
      } else {
        mapItr->second += energy;
      }
    }
  }
}


void CaloHitAnalyzer::analyze(int id, double recEnergy) {
  if(recEnergy > hitEnergyThreshold_) {
    map<int, double>::iterator mapItr = hitEnergySumMap_.find(id);
    if(mapItr == hitEnergySumMap_.end()) {
      ++noiseHits_;
    } else {
       // keep statistics of the rec energy / sim energy
      summary_.addEntry(recEnergy/mapItr->second);
    }
  }
}

