#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimMuon/MCTruth/interface/CSCHitAssociator.h"

CSCHitAssociator::Config::Config(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : linksTag_(conf.getParameter<edm::InputTag>("CSClinksTag")),
      linksToken_(iC.consumes(linksTag_)),
      geomToken_(iC.esConsumes()) {}

CSCHitAssociator::CSCHitAssociator(const edm::Event &event, const edm::EventSetup &setup, const Config &conf)
    : theConfig(conf), theDigiSimLinks(nullptr) {
  initEvent(event, setup);
}

void CSCHitAssociator::initEvent(const edm::Event &event, const edm::EventSetup &setup) {
  LogTrace("CSCHitAssociator") << "getting CSC Strip DigiSimLink collection - " << theConfig.linksTag_;
  theDigiSimLinks = &event.get(theConfig.linksToken_);

  // get CSC Geometry to use CSCLayer methods
  cscgeom = &setup.getData(theConfig.geomToken_);
}

std::vector<CSCHitAssociator::SimHitIdpr> CSCHitAssociator::associateCSCHitId(const CSCRecHit2D *cscrechit) const {
  std::vector<SimHitIdpr> simtrackids;

  unsigned int detId = cscrechit->geographicalId().rawId();
  int nchannels = cscrechit->nStrips();
  const CSCLayerGeometry *laygeom = cscgeom->layer(cscrechit->cscDetId())->geometry();

  DigiSimLinks::const_iterator layerLinks = theDigiSimLinks->find(detId);

  if (layerLinks != theDigiSimLinks->end()) {
    for (int idigi = 0; idigi < nchannels; ++idigi) {
      // strip and readout channel numbers may differ in ME1/1A
      int istrip = cscrechit->channels(idigi);
      int channel = laygeom->channel(istrip);

      for (LayerLinks::const_iterator link = layerLinks->begin(); link != layerLinks->end(); ++link) {
        int ch = static_cast<int>(link->channel());
        if (ch == channel) {
          SimHitIdpr currentId(link->SimTrackId(), link->eventId());
          if (find(simtrackids.begin(), simtrackids.end(), currentId) == simtrackids.end())
            simtrackids.push_back(currentId);
        }
      }
    }

  } else
    LogTrace("CSCHitAssociator") << "*** WARNING in CSCHitAssociator::associateCSCHitId - CSC layer " << detId
                                 << " has no DigiSimLinks !" << std::endl;

  return simtrackids;
}

std::vector<CSCHitAssociator::SimHitIdpr> CSCHitAssociator::associateHitId(const TrackingRecHit &hit) const {
  std::vector<SimHitIdpr> simtrackids;

  const TrackingRecHit *hitp = &hit;
  const CSCRecHit2D *cscrechit = dynamic_cast<const CSCRecHit2D *>(hitp);

  if (cscrechit) {
    unsigned int detId = cscrechit->geographicalId().rawId();
    int nchannels = cscrechit->nStrips();
    const CSCLayerGeometry *laygeom = cscgeom->layer(cscrechit->cscDetId())->geometry();

    DigiSimLinks::const_iterator layerLinks = theDigiSimLinks->find(detId);

    if (layerLinks != theDigiSimLinks->end()) {
      for (int idigi = 0; idigi < nchannels; ++idigi) {
        // strip and readout channel numbers may differ in ME1/1A
        int istrip = cscrechit->channels(idigi);
        int channel = laygeom->channel(istrip);

        for (LayerLinks::const_iterator link = layerLinks->begin(); link != layerLinks->end(); ++link) {
          int ch = static_cast<int>(link->channel());
          if (ch == channel) {
            SimHitIdpr currentId(link->SimTrackId(), link->eventId());
            if (find(simtrackids.begin(), simtrackids.end(), currentId) == simtrackids.end())
              simtrackids.push_back(currentId);
          }
        }
      }

    } else
      LogTrace("CSCHitAssociator") << "*** WARNING in CSCHitAssociator::associateHitId - CSC layer " << detId
                                   << " has no DigiSimLinks !" << std::endl;

  } else
    LogTrace("CSCHitAssociator") << "*** WARNING in CSCHitAssociator::associateHitId, null dynamic_cast "
                                    "!";

  return simtrackids;
}
