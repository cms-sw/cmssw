// -*- C++ -*-
// Class:      ME0ChamberMasker
//

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

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"
#include "CondFormats/RecoMuonObjects/interface/MuonSystemAging.h"
#include "CondFormats/DataRecord/interface/MuonSystemAgingRcd.h"
//
// class declaration
//

class ME0ChamberMasker : public edm::global::EDProducer<> {
public:
  explicit ME0ChamberMasker(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const bool me0Minus_;
  const bool me0Plus_;
  const edm::InputTag digiTag_;
  const edm::EDGetTokenT<ME0DigiPreRecoCollection> m_digiTag;
  const edm::EDPutTokenT<ME0DigiPreRecoCollection> m_putToken;
  const edm::ESGetToken<MuonSystemAging, MuonSystemAgingRcd> m_agingObjTag;
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
ME0ChamberMasker::ME0ChamberMasker(const edm::ParameterSet& iConfig)
    : me0Minus_(iConfig.getParameter<bool>("me0Minus")),
      me0Plus_(iConfig.getParameter<bool>("me0Plus")),
      digiTag_(iConfig.getParameter<edm::InputTag>("digiTag")),
      m_digiTag(consumes(digiTag_)),
      m_putToken(produces()),
      m_agingObjTag(esConsumes()) {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void ME0ChamberMasker::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  MuonSystemAging const& agingObj = iSetup.getData(m_agingObjTag);

  auto const& maskedME0IDs = agingObj.m_ME0ChambEffs;

  ME0DigiPreRecoCollection filteredDigis;

  if (!digiTag_.label().empty()) {
    ME0DigiPreRecoCollection const& me0Digis = iEvent.get(m_digiTag);

    for (const auto& me0LayerId : me0Digis) {
      auto chambId = me0LayerId.first.chamberId();

      bool keepDigi = (!me0Minus_ && chambId.region() < 0) || (!me0Plus_ && chambId.region() > 0);

      uint32_t rawId = chambId.rawId();
      if (keepDigi || maskedME0IDs.find(rawId) == maskedME0IDs.end()) {
        filteredDigis.put(me0LayerId.second, me0LayerId.first);
      }
    }
  }

  iEvent.emplace(m_putToken, std::move(filteredDigis));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void ME0ChamberMasker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("simMuonME0Digis"));
  desc.add<bool>("me0Minus", true);
  desc.add<bool>("me0Plus", true);
  descriptions.add("me0ChamberMasker", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ME0ChamberMasker);
