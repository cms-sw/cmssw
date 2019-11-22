// system include files
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <regex>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "CondFormats/RecoMuonObjects/interface/MuonSystemAging.h"
#include "CondFormats/DataRecord/interface/MuonSystemAgingRcd.h"

//
// class declaration
//

class GEMChamberMasker : public edm::stream::EDProducer<> {
public:
  explicit GEMChamberMasker(const edm::ParameterSet&);
  ~GEMChamberMasker() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::InputTag digiTag_;
  bool ge11Minus_;
  bool ge11Plus_;
  bool ge21Minus_;
  bool ge21Plus_;

  edm::EDGetTokenT<GEMDigiCollection> m_digiTag;
  std::map<unsigned int, float> m_maskedGEMIDs;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GEMChamberMasker::GEMChamberMasker(const edm::ParameterSet& iConfig)
    : digiTag_(iConfig.getParameter<edm::InputTag>("digiTag")),
      ge11Minus_(iConfig.getParameter<bool>("ge11Minus")),
      ge11Plus_(iConfig.getParameter<bool>("ge11Plus")),
      ge21Minus_(iConfig.getParameter<bool>("ge21Minus")),
      ge21Plus_(iConfig.getParameter<bool>("ge21Plus")) {
  m_digiTag = consumes<GEMDigiCollection>(digiTag_);
  produces<GEMDigiCollection>();
}

GEMChamberMasker::~GEMChamberMasker() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void GEMChamberMasker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::unique_ptr<GEMDigiCollection> filteredDigis(new GEMDigiCollection());

  if (!digiTag_.label().empty()) {
    edm::Handle<GEMDigiCollection> gemDigis;
    iEvent.getByToken(m_digiTag, gemDigis);

    for (const auto& gemLayerId : (*gemDigis)) {
      auto chambId = gemLayerId.first.chamberId();

      bool keepDigi = (!ge11Minus_ && chambId.station() == 1 && chambId.region() < 0) ||
                      (!ge11Plus_ && chambId.station() == 1 && chambId.region() > 0) ||
                      (!ge21Minus_ && chambId.station() == 2 && chambId.region() < 0) ||
                      (!ge21Plus_ && chambId.station() == 2 && chambId.region() > 0);

      uint32_t rawId = chambId.rawId();
      if (keepDigi || m_maskedGEMIDs.find(rawId) == m_maskedGEMIDs.end()) {
        filteredDigis->put(gemLayerId.second, gemLayerId.first);
      }
    }
  }

  iEvent.put(std::move(filteredDigis));
}

// ------------ method called when starting to processes a run  ------------

void GEMChamberMasker::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  edm::ESHandle<MuonSystemAging> agingObj;
  iSetup.get<MuonSystemAgingRcd>().get(agingObj);

  m_maskedGEMIDs = agingObj->m_GEMChambEffs;
}

// ------------ method called when ending the processing of a run  ------------

void GEMChamberMasker::endRun(edm::Run const&, edm::EventSetup const&) {}

void GEMChamberMasker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("simMuonGEMDigis"));
  desc.add<bool>("ge11Minus", true);
  desc.add<bool>("ge11Plus", true);
  desc.add<bool>("ge21Minus", true);
  desc.add<bool>("ge21Plus", true);

  descriptions.add("gemChamberMasker", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMChamberMasker);
