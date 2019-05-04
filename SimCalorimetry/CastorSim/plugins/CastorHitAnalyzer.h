#ifndef CastorSim_CastorHitAnalyzer_h
#define CastorSim_CastorHitAnalyzer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "SimCalorimetry/CastorSim/src/CastorHitFilter.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"
#include <string>

/** Compares RecHits to SimHit

  \Author P. Katsas, Univ. of Athens
*/

class CastorHitAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CastorHitAnalyzer(edm::ParameterSet const &conf);
  void analyze(edm::Event const &e, edm::EventSetup const &c) override;

private:
  std::string hitReadoutName_;
  CastorSimParameterMap simParameterMap_;
  CastorHitFilter castorFilter_;
  CaloHitAnalyzer castorAnalyzer_;
  edm::InputTag castorRecHitCollectionTag_;
};

#endif
