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
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
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

class DTChamberMasker : public edm::stream::EDProducer<> {
public:
  explicit DTChamberMasker(const edm::ParameterSet &);
  ~DTChamberMasker() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  void beginRun(edm::Run const &, edm::EventSetup const &) override;

  void createMaskedChamberCollection(edm::ESHandle<DTGeometry> &);

  // ----------member data ---------------------------

  edm::EDGetTokenT<DTDigiCollection> m_digiToken;
  std::map<unsigned int, float> m_ChEffs;
};

//
// constants, enums and typedefs
//

//
// constructors and destructor
//
DTChamberMasker::DTChamberMasker(const edm::ParameterSet &iConfig)
    : m_digiToken(consumes<DTDigiCollection>(iConfig.getParameter<edm::InputTag>("digiTag"))) {
  produces<DTDigiCollection>();
}

DTChamberMasker::~DTChamberMasker() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void DTChamberMasker::produce(edm::Event &event, const edm::EventSetup &conditions) {
  edm::Service<edm::RandomNumberGenerator> randGenService;
  CLHEP::HepRandomEngine &randGen = randGenService->getEngine(event.streamID());

  std::unique_ptr<DTDigiCollection> filteredDigis(new DTDigiCollection());

  if (!m_digiToken.isUninitialized()) {
    edm::Handle<DTDigiCollection> dtDigis;
    event.getByToken(m_digiToken, dtDigis);

    for (const auto &dtLayerId : (*dtDigis)) {
      uint32_t rawId = (dtLayerId.first).chamberId().rawId();
      auto chEffIt = m_ChEffs.find(rawId);

      if (chEffIt == m_ChEffs.end() || randGen.flat() <= chEffIt->second)
        filteredDigis->put(dtLayerId.second, dtLayerId.first);
    }
  }

  event.put(std::move(filteredDigis));
}

// ------------ method called when starting to processes a run  ------------
void DTChamberMasker::beginRun(edm::Run const &run, edm::EventSetup const &iSetup) {
  m_ChEffs.clear();

  edm::ESHandle<MuonSystemAging> agingObj;
  iSetup.get<MuonSystemAgingRcd>().get(agingObj);

  m_ChEffs = agingObj->m_DTChambEffs;
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
