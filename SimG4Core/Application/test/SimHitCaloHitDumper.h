#ifndef SimHitCaloHitDumper_H
#define SimHitCaloHitDumper_H
// 
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class SimHitCaloHitDumper : public edm::EDAnalyzer{
   public:
  explicit SimHitCaloHitDumper( const edm::ParameterSet& ){};
  virtual ~SimHitCaloHitDumper() {};
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  virtual void beginJob( const edm::EventSetup& ){};
  virtual void endJob(){};

};

#endif
