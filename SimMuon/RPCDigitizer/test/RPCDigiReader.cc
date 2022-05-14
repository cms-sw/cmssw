/** \class RPCDigiReader
 *  Analyse the RPC digitizer (derived from R. Bellan DTDigiReader. 
 *  
 *  \authors: M. Maggi -- INFN Bari
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include <map>
#include <set>

#include "DataFormats/Common/interface/DetSet.h"

#include <iostream>

class RPCDigiReader : public edm::one::EDAnalyzer<> {
public:
  explicit RPCDigiReader(const edm::ParameterSet& pset)
      : label_(pset.getUntrackedParameter<std::string>("label")),
        tokGeom_(esConsumes<RPCGeometry, MuonGeometryRecord>()),
        digiToken_(consumes<RPCDigiCollection>(edm::InputTag(label_))),
        hitsToken_(consumes<edm::PSimHitContainer>(edm::InputTag("g4SimHits", "MuonRPCHits"))),
        linkToken_(consumes<edm::DetSetVector<RPCDigiSimLink> >(edm::InputTag("muonRPCDigis", "RPCDigiSimLink"))) {}

  ~RPCDigiReader() override = default;

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override {
    edm::LogVerbatim("RPCDump") << "--- Run: " << event.id().run() << " Event: " << event.id().event();

    const auto& rpcDigis = event.getHandle(digiToken_);
    const auto& simHits = event.getHandle(hitsToken_);
    const auto& thelinkDigis = event.getHandle(linkToken_);

    const auto& pDD = eventSetup.getHandle(tokGeom_);

    RPCDigiCollection::DigiRangeIterator detUnitIt;
    for (detUnitIt = rpcDigis->begin(); detUnitIt != rpcDigis->end(); ++detUnitIt) {
      const RPCDetId& id = (*detUnitIt).first;
      const RPCRoll* roll = dynamic_cast<const RPCRoll*>(pDD->roll(id));
      const RPCDigiCollection::Range& range = (*detUnitIt).second;

      //     if(id.rawId() != 637567293) continue;

      // RPCDetId print-out
      edm::LogVerbatim("RPCDump") << "--------------";
      edm::LogVerbatim("RPCDump") << "id: " << id.rawId() << " number of strip " << roll->nstrips();

      // Loop over the digis of this DetUnit
      for (RPCDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
        edm::LogVerbatim("RPCDump") << " digi " << *digiIt;
        if (digiIt->strip() < 1 || digiIt->strip() > roll->nstrips()) {
          edm::LogVerbatim("RPCDump") << " XXXXXXXXXXXXX Problemt with " << id;
        }
        for (std::vector<PSimHit>::const_iterator simHit = simHits->begin(); simHit != simHits->end(); simHit++) {
          RPCDetId rpcId((*simHit).detUnitId());
          if (rpcId == id && abs((*simHit).particleType()) == 13) {
            edm::LogVerbatim("RPCDump") << "entry: " << (*simHit).entryPoint() << "\nexit: " << (*simHit).exitPoint()
                                        << "\nTOF: " << (*simHit).timeOfFlight();
          }
        }
      }  // for digis in layer
    }    // for layers

    for (edm::DetSetVector<RPCDigiSimLink>::const_iterator itlink = thelinkDigis->begin();
         itlink != thelinkDigis->end();
         itlink++) {
      for (edm::DetSet<RPCDigiSimLink>::const_iterator digi_iter = itlink->data.begin();
           digi_iter != itlink->data.end();
           ++digi_iter) {
        int ev = digi_iter->getEventId().event();
        int detid = digi_iter->getDetUnitId();
        float xpos = digi_iter->getEntryPoint().x();
        int strip = digi_iter->getStrip();
        int bx = digi_iter->getBx();

        edm::LogVerbatim("RPCDump") << "DetUnit: " << detid << "  "
                                    << "Event ID: " << ev << "  "
                                    << "Pos X: " << xpos << "  "
                                    << "Strip: " << strip << "  "
                                    << "Bx: " << bx;
      }
    }

    edm::LogVerbatim("RPCDump") << "--------------";
  }

private:
  const std::string label_;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> tokGeom_;
  const edm::EDGetTokenT<RPCDigiCollection> digiToken_;
  const edm::EDGetTokenT<edm::PSimHitContainer> hitsToken_;
  const edm::EDGetTokenT<edm::DetSetVector<RPCDigiSimLink> > linkToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCDigiReader);
