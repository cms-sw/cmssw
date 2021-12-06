/**_________________________________________________________________
   class:   BeamSpotFromDB.h
   package: RecoVertex/BeamSpotProducer
   
   author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
________________________________________________________________**/

// C++ standard
#include <string>

// CMS
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

class BeamSpotFromDB : public edm::one::EDAnalyzer<> {
public:
  explicit BeamSpotFromDB(const edm::ParameterSet&);
  ~BeamSpotFromDB() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> m_beamToken;
};

BeamSpotFromDB::BeamSpotFromDB(const edm::ParameterSet& iConfig)
    : m_beamToken(esConsumes<BeamSpotObjects, BeamSpotObjectsRcd>()) {}

BeamSpotFromDB::~BeamSpotFromDB() = default;

void BeamSpotFromDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const BeamSpotObjects* mybeamspot = &iSetup.getData(m_beamToken);
  edm::LogPrint("BeamSpotFromDB") << " for runs: " << iEvent.id().run() << " - " << iEvent.id().run();
  //edm::LogPrint("BeamSpotFromDB") << iEvent.getRun().beginTime().value();
  //edm::LogPrint("BeamSpotFromDB") << iEvent.time().value();
  edm::LogPrint("BeamSpotFromDB") << *mybeamspot;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotFromDB);
