#ifndef SimHitCaloHitDumper_H
#define SimHitCaloHitDumper_H
// 
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SimHitCaloHitDumper : public edm::EDAnalyzer{
 public:
  explicit SimHitCaloHitDumper( const edm::ParameterSet& iConfig ):
    processName(iConfig.getParameter<std::string>("processName")){};
  ~SimHitCaloHitDumper() override {};
  
  void analyze( const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override{};
  void endJob() override{};
  
 private:
  std::string processName;

};

#endif
