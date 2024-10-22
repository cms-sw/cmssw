// system include files
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <regex>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
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

class GEMChamberMasker : public edm::global::EDProducer<> {
public:
  explicit GEMChamberMasker(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const edm::InputTag digiTag_;
  const bool ge11Minus_;
  const bool ge11Plus_;
  const bool ge21Minus_;
  const bool ge21Plus_;

  const edm::EDGetTokenT<GEMDigiCollection> m_digiTag;
  const edm::EDPutTokenT<GEMDigiCollection> m_putToken;
  const edm::ESGetToken<MuonSystemAging, MuonSystemAgingRcd> m_agingObj;
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
      ge21Plus_(iConfig.getParameter<bool>("ge21Plus")),
      m_digiTag(consumes(digiTag_)),
      m_putToken(produces()),
      m_agingObj(esConsumes()) {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void GEMChamberMasker::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  GEMDigiCollection filteredDigis;

  auto const& agingObj = iSetup.getData(m_agingObj);

  auto const& maskedGEMIDs = agingObj.m_GEMChambEffs;

  if (!digiTag_.label().empty()) {
    GEMDigiCollection const& gemDigis = iEvent.get(m_digiTag);

    for (const auto& gemLayerId : gemDigis) {
      auto chambId = gemLayerId.first.chamberId();

      bool keepDigi = (!ge11Minus_ && chambId.station() == 1 && chambId.region() < 0) ||
                      (!ge11Plus_ && chambId.station() == 1 && chambId.region() > 0) ||
                      (!ge21Minus_ && chambId.station() == 2 && chambId.region() < 0) ||
                      (!ge21Plus_ && chambId.station() == 2 && chambId.region() > 0);

      uint32_t rawId = chambId.rawId();
      if (keepDigi || maskedGEMIDs.find(rawId) == maskedGEMIDs.end()) {
        filteredDigis.put(gemLayerId.second, gemLayerId.first);
      }
    }
  }

  iEvent.emplace(m_putToken, std::move(filteredDigis));
}

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
