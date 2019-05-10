// -*- C++ -*-
//
// Package:    SimMuon/CSCDigitizer
// Class:      CSCChamberMasker
//
/**\class CSCChamberMasker CSCChamberMasker.cc
 SimMuon/CSCDigitizer/plugins/CSCChamberMasker.cc

 Description: Class to mask CSC digis on a chamber by chamber basis

*/
//
// Original Author:  Nick J. Amin
//         Created:  Mon, 27 Feb 2017 15:12:51 GMT
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

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"

#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

#include "CondFormats/DataRecord/interface/MuonSystemAgingRcd.h"
#include "CondFormats/RecoMuonObjects/interface/MuonSystemAging.h"

#include "CLHEP/Random/RandomEngine.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

//
// class declaration
//

class CSCChamberMasker : public edm::stream::EDProducer<> {
public:
  explicit CSCChamberMasker(const edm::ParameterSet &);
  ~CSCChamberMasker() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  void beginRun(edm::Run const &, edm::EventSetup const &) override;

  void createMaskedChamberCollection(edm::ESHandle<CSCGeometry> &);

  template <typename T, typename C = MuonDigiCollection<CSCDetId, T>>
  void ageDigis(edm::Event &event,
                edm::EDGetTokenT<C> &digiToken,
                CLHEP::HepRandomEngine &randGen,
                std::unique_ptr<C> &filteredDigis);

  template <typename T, typename C = MuonDigiCollection<CSCDetId, T>>
  void copyDigis(edm::Event &event, edm::EDGetTokenT<C> &digiToken, std::unique_ptr<C> &filteredDigis);

  // ----------member data ---------------------------

  edm::EDGetTokenT<CSCStripDigiCollection> m_stripDigiToken;
  edm::EDGetTokenT<CSCWireDigiCollection> m_wireDigiToken;
  edm::EDGetTokenT<CSCCLCTDigiCollection> m_clctDigiToken;
  edm::EDGetTokenT<CSCALCTDigiCollection> m_alctDigiToken;
  std::map<CSCDetId, std::pair<unsigned int, float>> m_CSCEffs;
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
CSCChamberMasker::CSCChamberMasker(const edm::ParameterSet &iConfig)
    : m_stripDigiToken(consumes<CSCStripDigiCollection>(iConfig.getParameter<edm::InputTag>("stripDigiTag"))),
      m_wireDigiToken(consumes<CSCWireDigiCollection>(iConfig.getParameter<edm::InputTag>("wireDigiTag"))),
      m_clctDigiToken(consumes<CSCCLCTDigiCollection>(iConfig.getParameter<edm::InputTag>("clctDigiTag"))),
      m_alctDigiToken(consumes<CSCALCTDigiCollection>(iConfig.getParameter<edm::InputTag>("alctDigiTag"))) {
  produces<CSCStripDigiCollection>("MuonCSCStripDigi");
  produces<CSCWireDigiCollection>("MuonCSCWireDigi");
  produces<CSCCLCTDigiCollection>("MuonCSCCLCTDigi");
  produces<CSCALCTDigiCollection>("MuonCSCALCTDigi");
}

CSCChamberMasker::~CSCChamberMasker() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void CSCChamberMasker::produce(edm::Event &event, const edm::EventSetup &conditions) {
  edm::Service<edm::RandomNumberGenerator> randGenService;
  CLHEP::HepRandomEngine &randGen = randGenService->getEngine(event.streamID());

  std::unique_ptr<CSCStripDigiCollection> filteredStripDigis(new CSCStripDigiCollection());
  std::unique_ptr<CSCWireDigiCollection> filteredWireDigis(new CSCWireDigiCollection());
  std::unique_ptr<CSCCLCTDigiCollection> filteredCLCTDigis(new CSCCLCTDigiCollection());
  std::unique_ptr<CSCALCTDigiCollection> filteredALCTDigis(new CSCALCTDigiCollection());

  // Handle wire and strip digis
  ageDigis<CSCStripDigi>(event, m_stripDigiToken, randGen, filteredStripDigis);
  ageDigis<CSCWireDigi>(event, m_wireDigiToken, randGen, filteredWireDigis);

  // Don't touch CLCT or ALCT digis
  copyDigis<CSCCLCTDigi>(event, m_clctDigiToken, filteredCLCTDigis);
  copyDigis<CSCALCTDigi>(event, m_alctDigiToken, filteredALCTDigis);

  event.put(std::move(filteredStripDigis), "MuonCSCStripDigi");
  event.put(std::move(filteredWireDigis), "MuonCSCWireDigi");
  event.put(std::move(filteredCLCTDigis), "MuonCSCCLCTDigi");
  event.put(std::move(filteredALCTDigis), "MuonCSCALCTDigi");
}

// ------------ method called to copy digis into aged collection  ------------
template <typename T, typename C>
void CSCChamberMasker::copyDigis(edm::Event &event, edm::EDGetTokenT<C> &digiToken, std::unique_ptr<C> &filteredDigis) {
  if (!digiToken.isUninitialized()) {
    edm::Handle<C> digis;
    event.getByToken(digiToken, digis);
    for (const auto &j : (*digis)) {
      auto digiItr = j.second.first;
      auto last = j.second.second;

      CSCDetId const cscDetId = j.first;

      for (; digiItr != last; ++digiItr) {
        filteredDigis->insertDigi(cscDetId, *digiItr);
      }
    }
  }
}

// ------------ method aging digis------------
template <typename T, typename C>
void CSCChamberMasker::ageDigis(edm::Event &event,
                                edm::EDGetTokenT<C> &digiToken,
                                CLHEP::HepRandomEngine &randGen,
                                std::unique_ptr<C> &filteredDigis) {
  if (!digiToken.isUninitialized()) {
    edm::Handle<C> digis;
    event.getByToken(digiToken, digis);

    for (const auto &j : (*digis)) {
      auto digiItr = j.second.first;
      auto last = j.second.second;

      CSCDetId const cscDetId = j.first;

      // Since lookups are chamber-centric, make new DetId with layer=0
      CSCDetId chId = CSCDetId(cscDetId.endcap(), cscDetId.station(), cscDetId.ring(), cscDetId.chamber(), 0);

      for (; digiItr != last; ++digiItr) {
        auto chEffIt = m_CSCEffs.find(chId);

        if (chEffIt != m_CSCEffs.end()) {
          std::pair<unsigned int, float> typeEff = chEffIt->second;
          int type = typeEff.first % 10;   // second digit gives type of inefficiency
          int layer = typeEff.first / 10;  // first digit gives layer (0 = chamber level)

          bool doRandomize = false;
          if (((std::is_same<T, CSCStripDigi>::value && type == EFF_WIRES) ||
               (std::is_same<T, CSCWireDigi>::value && type == EFF_STRIPS) || type == EFF_CHAMBER) &&
              (layer == 0 || cscDetId.layer() == layer))
            doRandomize = true;

          if (!doRandomize || (randGen.flat() <= typeEff.second)) {
            filteredDigis->insertDigi(cscDetId, *digiItr);
          }
        }
      }
    }
  }
}

// ------------ method called when starting to processes a run  ------------
void CSCChamberMasker::beginRun(edm::Run const &run, edm::EventSetup const &iSetup) {
  m_CSCEffs.clear();

  edm::ESHandle<CSCGeometry> cscGeom;
  iSetup.get<MuonGeometryRecord>().get(cscGeom);

  edm::ESHandle<MuonSystemAging> agingObj;
  iSetup.get<MuonSystemAgingRcd>().get(agingObj);

  const auto chambers = cscGeom->chambers();

  for (const auto *ch : chambers) {
    CSCDetId chId = ch->id();
    unsigned int rawId = chId.rawIdMaker(chId.endcap(), chId.station(), chId.ring(), chId.chamber(), 0);
    float eff = 1.;
    int type = 0;
    for (auto &agingPair : agingObj->m_CSCChambEffs) {
      if (agingPair.first != rawId)
        continue;

      type = agingPair.second.first;
      eff = agingPair.second.second;
      m_CSCEffs[chId] = std::make_pair(type, eff);
      break;
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the
// module  ------------
void CSCChamberMasker::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("stripDigiTag", edm::InputTag("simMuonCSCDigis:MuonCSCStripDigi"));
  desc.add<edm::InputTag>("wireDigiTag", edm::InputTag("simMuonCSCDigis:MuonCSCWireDigi"));
  desc.add<edm::InputTag>("comparatorDigiTag", edm::InputTag("simMuonCSCDigis:MuonCSCComparatorDigi"));
  desc.add<edm::InputTag>("rpcDigiTag", edm::InputTag("simMuonCSCDigis:MuonCSCRPCDigi"));
  desc.add<edm::InputTag>("alctDigiTag", edm::InputTag("simMuonCSCDigis:MuonCSCALCTDigi"));
  desc.add<edm::InputTag>("clctDigiTag", edm::InputTag("simMuonCSCDigis:MuonCSCCLCTDigi"));
  descriptions.add("cscChamberMasker", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(CSCChamberMasker);
