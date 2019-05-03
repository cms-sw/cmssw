#ifndef CastorSim_CastorDigiAnalyzer_h
#define CastorSim_CastorDigiAnalyzer_h

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "SimCalorimetry/CastorSim/plugins/CastorDigiStatistics.h"
#include "SimCalorimetry/CastorSim/src/CastorHitFilter.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"
#include <string>

/**  Castor digis
 Author: Panos Katsas
*/

class CastorDigiAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CastorDigiAnalyzer(edm::ParameterSet const &conf);
  void analyze(edm::Event const &e, edm::EventSetup const &c) override;

private:
  std::string hitReadoutName_;
  CastorSimParameterMap simParameterMap_;
  CastorHitFilter castorFilter_;
  CaloHitAnalyzer castorHitAnalyzer_;
  CastorDigiStatistics castorDigiStatistics_;
  edm::InputTag castorDigiCollectionTag_;
};

#endif
