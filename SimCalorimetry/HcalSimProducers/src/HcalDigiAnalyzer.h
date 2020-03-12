#ifndef HcalSimProducers_HcalDigiAnalyzer_h
#define HcalSimProducers_HcalDigiAnalyzer_h

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalDigiStatistics.h"
#include <string>

/** Studies Hcal digis

  \Author Rick Wilkinson, Caltech
*/

class HcalDigiAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalDigiAnalyzer(edm::ParameterSet const &conf);
  void analyze(edm::Event const &e, edm::EventSetup const &c) override;

private:
  std::string hitReadoutName_;
  HcalSimParameterMap simParameterMap_;
  HBHEHitFilter hbheFilter_;
  HOHitFilter hoFilter_;
  HFHitFilter hfFilter_;
  ZDCHitFilter zdcFilter_;
  CaloHitAnalyzer hbheHitAnalyzer_;
  CaloHitAnalyzer hoHitAnalyzer_;
  CaloHitAnalyzer hfHitAnalyzer_;
  CaloHitAnalyzer zdcHitAnalyzer_;
  HcalDigiStatistics hbheDigiStatistics_;
  HcalDigiStatistics hoDigiStatistics_;
  HcalDigiStatistics hfDigiStatistics_;
  HcalDigiStatistics zdcDigiStatistics_;

  edm::InputTag hbheDigiCollectionTag_;
  edm::InputTag hoDigiCollectionTag_;
  edm::InputTag hfDigiCollectionTag_;
};

#endif
