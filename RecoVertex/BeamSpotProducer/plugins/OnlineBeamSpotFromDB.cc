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
class MyHelper {
  public:
    MyHelper(edm::ParameterSet const& iPS, edm::ConsumesCollector && iC, edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd>& bsToken_){    
       bsToken_ =iC.esConsumes<BeamSpotObjects, BeamSpotTransientObjectsRcd>();
    }
};

class OnlineBeamSpotFromDB : public edm::one::EDAnalyzer<> {
public:
  explicit OnlineBeamSpotFromDB(const edm::ParameterSet& iConfig);
  ~OnlineBeamSpotFromDB() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& desc);
  edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd> bsToken_;
  

private:  
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
};

OnlineBeamSpotFromDB::OnlineBeamSpotFromDB(const edm::ParameterSet& iConfig) {
  MyHelper m_helper(iConfig, consumesCollector(),bsToken_); 
 
}

OnlineBeamSpotFromDB::~OnlineBeamSpotFromDB() {}

void OnlineBeamSpotFromDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //edm::ESHandle<BeamSpotObjects> beamGThandle;

  auto const& mybeamspot = iSetup.getData(bsToken_);
  //iSetup.get<BeamSpotTransientObjectsRcd>().get(beamhandle);
  //const BeamSpotObjects* mybeamspot = beamhandle.product();
  //iSetup.get<BeamSpotObjectsRcd>().get(beamGThandle);
  //const BeamSpotObjects* myGTbeamspot = beamGThandle.product();

  edm::LogInfo("Run numver: ") << iEvent.id().run();
  edm::LogInfo("beamspot from HLT ") << mybeamspot;
  //edm::LogInfo("beamspot from GT ")<<*myGTbeamspot;
}
void OnlineBeamSpotFromDB::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription dsc;
  desc.addWithDefaultLabel(dsc);
}
void OnlineBeamSpotFromDB::beginJob() {}

void OnlineBeamSpotFromDB::endJob() {}



//define this as a plug-in
DEFINE_FWK_MODULE(OnlineBeamSpotFromDB);
