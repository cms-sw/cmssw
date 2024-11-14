// -*- C++ -*-
//
// Package:    SimMuon/DTDigitizer
// Class:      DTChamberMasker
//
/**\class DTChamberMasker DTChamberMasker.cc
 SimMuon/DTDigitizer/plugins/DTChamberMasker.cc

 Description: Class to mask DT digis on a chamber by chamber basis

*/
//
// Original Author:  Carlo Battilana
//         Created:  Sun, 11 Jan 2015 15:12:51 GMT
//
//

// system include files
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLinkCollection.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "CondFormats/DataRecord/interface/MuonSystemAgingRcd.h"
#include "CondFormats/RecoMuonObjects/interface/MuonSystemAging.h"

#include "CLHEP/Random/RandomEngine.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

//
// class declaration
//

class DTChamberMasker : public edm::global::EDProducer<> {
public:
  explicit DTChamberMasker(const edm::ParameterSet &);

  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  void createMaskedChamberCollection(edm::ESHandle<DTGeometry> &);

  // ----------member data ---------------------------

  const edm::EDGetTokenT<DTDigiCollection> m_digiTokenR;
  const edm::EDGetTokenT<DTDigiSimLinkCollection> m_linkTokenR;
  const edm::EDPutTokenT<DTDigiCollection> m_digiTokenP;
  const edm::EDPutTokenT<DTDigiSimLinkCollection> m_linkTokenP;
  const edm::ESGetToken<MuonSystemAging, MuonSystemAgingRcd> m_agingObjToken;
};

//
// constants, enums and typedefs
//

//
// constructors and destructor
//
DTChamberMasker::DTChamberMasker(const edm::ParameterSet &iConfig)
    : m_digiTokenR(consumes(iConfig.getParameter<edm::InputTag>("digiTag"))),
      m_linkTokenR(consumes(iConfig.getParameter<edm::InputTag>("digiTag"))),
      m_digiTokenP(produces()),
      m_linkTokenP(produces()),
      m_agingObjToken(esConsumes()) {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void DTChamberMasker::produce(edm::StreamID, edm::Event &event, const edm::EventSetup &conditions) const {
  edm::Service<edm::RandomNumberGenerator> randGenService;
  CLHEP::HepRandomEngine &randGen = randGenService->getEngine(event.streamID());

  MuonSystemAging const &agingObj = conditions.getData(m_agingObjToken);

  auto const &chEffs = agingObj.m_DTChambEffs;

  DTDigiCollection filteredDigis;

  if (!m_digiTokenR.isUninitialized()) {
    edm::Handle<DTDigiCollection> dtDigis;
    event.getByToken(m_digiTokenR, dtDigis);

    for (const auto &dtLayerId : (*dtDigis)) {
      uint32_t rawId = (dtLayerId.first).chamberId().rawId();
      auto chEffIt = chEffs.find(rawId);

      if (chEffIt == chEffs.end() || randGen.flat() <= chEffIt->second)
        filteredDigis.put(dtLayerId.second, dtLayerId.first);
    }
  }

  DTDigiSimLinkCollection linksCopy;

  if (!m_linkTokenR.isUninitialized()) {
    edm::Handle<DTDigiSimLinkCollection> links;
    event.getByToken(m_linkTokenR, links);
    linksCopy = (*links);
  }

  event.emplace(m_digiTokenP, std::move(filteredDigis));
  event.emplace(m_linkTokenP, std::move(linksCopy));
}

// ------------ method fills 'descriptions' with the allowed parameters for the
// module  ------------
void DTChamberMasker::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("simMuonDTDigis"));
  descriptions.add("dtChamberMasker", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(DTChamberMasker);
