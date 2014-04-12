#ifndef HcalSimProducers_HcalHitAnalyzer_h
#define HcalSimProducers_HcalHitAnalyzer_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HBHEHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HOHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"
#include <string>

/** Compares HCAL RecHits to SimHit

  \Author Rick Wilkinson, Caltech
*/


class HcalHitAnalyzer : public edm::EDAnalyzer
{
public:

  explicit HcalHitAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);


private:
  HcalSimParameterMap simParameterMap_;
  HBHEHitFilter hbheFilter_;
  HOHitFilter hoFilter_;
  HFHitFilter hfFilter_;
  ZDCHitFilter zdcFilter_;
  CaloHitAnalyzer hbheAnalyzer_;
  CaloHitAnalyzer hoAnalyzer_;
  CaloHitAnalyzer hfAnalyzer_;
  CaloHitAnalyzer zdcAnalyzer_;

  edm::InputTag hbheRecHitCollectionTag_;
  edm::InputTag hoRecHitCollectionTag_;
  edm::InputTag hfRecHitCollectionTag_;
};

#endif
