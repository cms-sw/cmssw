#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigiCommon.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace phase2trackerdigi {
  unsigned int getLayerNumber(unsigned int& detid, const TrackerTopology* topo) {
    unsigned int layer = 999;
    const DetId theDetId(detid);
    
    if (theDetId.det() == DetId::Tracker) {
      if (theDetId.subdetId() == PixelSubdetector::PixelBarrel) {
	layer = topo->pxbLayer(detid);
      } else if (theDetId.subdetId() == PixelSubdetector::PixelEndcap) {
	layer = 100 * topo->pxfSide(detid)  + topo->pxfDisk(detid);
      } else {
	edm::LogInfo("Phase2TrackerDigiCommon") << ">>> Invalid subdetId()  " ;
      }
    }
    return layer;
  }

  unsigned int getLayerNumber(unsigned int& detid) {
    unsigned int layer = 999;
    DetId theDetId(detid);
    if (theDetId.det() == DetId::Tracker) {
      if (theDetId.subdetId() == PixelSubdetector::PixelBarrel) {
	PXBDetId pb_detId = PXBDetId(detid);
	layer = pb_detId.layer();
      } else if (theDetId.subdetId() == PixelSubdetector::PixelEndcap) {
	PXFDetId pf_detId = PXFDetId(detid);
	layer = 100*pf_detId.side() + pf_detId.disk();
      } else edm::LogInfo("Phase2TrackerDigiCommon") << ">>> Invalid subdetId()  " ;
    }
    return layer;
  }
}    
