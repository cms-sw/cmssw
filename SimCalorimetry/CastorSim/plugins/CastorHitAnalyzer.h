#ifndef CastorSim_CastorHitAnalyzer_h
#define CastorSim_CastorHitAnalyzer_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"
#include "SimCalorimetry/CastorSim/src/CastorHitFilter.h"
#include <string>

/** Compares RecHits to SimHit

  \Author P. Katsas, Univ. of Athens
*/

class CastorHitAnalyzer : public edm::EDAnalyzer
{
public:

  explicit CastorHitAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);


private:
  std::string hitReadoutName_;
  CastorSimParameterMap simParameterMap_;
  CastorHitFilter castorFilter_;
  CaloHitAnalyzer castorAnalyzer_;
  edm::InputTag castorRecHitCollectionTag_;
};

#endif


