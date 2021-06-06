/**_________________________________________________________________
   class:   OnlineBeamSpotFromDB.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include <string>

class OnlineBeamSpotFromDB : public edm::one::EDAnalyzer<> {
public:
  explicit OnlineBeamSpotFromDB(const edm::ParameterSet& iConfig);
  static void fillDescriptions(edm::ConfigurationDescriptions& desc);
  edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd> bsToken_;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
};

OnlineBeamSpotFromDB::OnlineBeamSpotFromDB(const edm::ParameterSet& iConfig)
    : bsToken_(esConsumes<BeamSpotObjects, BeamSpotTransientObjectsRcd>()) {}

void OnlineBeamSpotFromDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& mybeamspot = iSetup.getData(bsToken_);

  edm::LogInfo("Run numver: ") << iEvent.id().run();
  edm::LogInfo("beamspot from HLT ") << mybeamspot;
}
void OnlineBeamSpotFromDB::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription dsc;
  desc.addWithDefaultLabel(dsc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(OnlineBeamSpotFromDB);
