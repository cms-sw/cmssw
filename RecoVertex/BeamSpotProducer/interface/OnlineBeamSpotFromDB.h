#ifndef BeamSpotProducer_OnlineBeamSpotFromDB_h
#define BeamSpotProducer_OnlineBeamSpotFromDB_h

/**_________________________________________________________________
   class:   OnlineBeamSpotFromDB.h
   package: RecoVertex/BeamSpotProducer
   


 author: 


________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class OnlineBeamSpotFromDB : public edm::EDAnalyzer {
public:
  explicit OnlineBeamSpotFromDB(const edm::ParameterSet&);
  ~OnlineBeamSpotFromDB() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
};

#endif
