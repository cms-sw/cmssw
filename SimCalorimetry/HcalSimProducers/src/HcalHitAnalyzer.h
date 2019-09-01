#ifndef HcalSimProducers_HcalHitAnalyzer_h
#define HcalSimProducers_HcalHitAnalyzer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"
#include <string>

/** Compares HCAL RecHits to SimHit

  \Author Rick Wilkinson, Caltech
*/

class HcalHitAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalHitAnalyzer(edm::ParameterSet const &conf);
  void analyze(edm::Event const &e, edm::EventSetup const &c) override;

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
