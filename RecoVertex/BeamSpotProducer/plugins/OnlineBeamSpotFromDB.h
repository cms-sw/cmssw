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
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
namespace edm {
  class ParameterSetDescription;

}
class OnlineBeamSpotFromDB : public edm::one::EDAnalyzer<> {
public:
  explicit OnlineBeamSpotFromDB(const edm::ParameterSet&);
  ~OnlineBeamSpotFromDB() override;
  static void fillDescription(edm::ConfigurationDescriptions& desc);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
};

#endif
