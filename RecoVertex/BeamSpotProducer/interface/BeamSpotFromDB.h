#ifndef BeamSpotProducer_BeamSpotFromDB_h
#define BeamSpotProducer_BeamSpotFromDB_h

/**_________________________________________________________________
   class:   BeamSpotFromDB.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class BeamSpotFromDB : public edm::EDAnalyzer {
 public:
  explicit BeamSpotFromDB(const edm::ParameterSet&);
  ~BeamSpotFromDB() override;

 private:
  void beginJob() override ;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;


};

#endif
