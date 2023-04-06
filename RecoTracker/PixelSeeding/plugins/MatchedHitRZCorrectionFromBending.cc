#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/ThirdHitPredictionFromCircle.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "MatchedHitRZCorrectionFromBending.h"

// Note: only the TIB layers seem to cause a significant effect,
//       so only z correction is implemented

MatchedHitRZCorrectionFromBending::MatchedHitRZCorrectionFromBending(DetId detId, const TrackerTopology *tTopo)
    : rFixup(nullptr), zFixup(nullptr) {
  if (detId.subdetId() == SiStripDetId::TIB && tTopo->tibIsDoubleSide(detId))
    zFixup = tibMatchedHitZFixup;
}

MatchedHitRZCorrectionFromBending::MatchedHitRZCorrectionFromBending(const DetLayer *layer,
                                                                     const TrackerTopology *tTopo)
    : rFixup(nullptr), zFixup(nullptr) {
  if (layer->subDetector() == GeomDetEnumerators::TIB) {
    const GeometricSearchDet *tibLayer = layer;

    if (tTopo->tibIsDoubleSide(tibLayer->basicComponents()[0]->geographicalId()))
      zFixup = tibMatchedHitZFixup;
  }
}

double MatchedHitRZCorrectionFromBending::tibMatchedHitZFixup(const ThirdHitPredictionFromCircle &pred,
                                                              double curvature,
                                                              double r,
                                                              const TrackingRecHit &hit,
                                                              const TrackerTopology *tTopo) {
  // the factors for [ TIB1=0, TIB2=1 ] [ inner string=0, outer string=1 ]
  static const double factors[2][2] = {{-2.4, 2.4}, {2.4, -2.4}};

  unsigned int layer = tTopo->tibLayer(hit.det()->geographicalId()) - 1;
  unsigned int string = !tTopo->tibIsInternalString(hit.det()->geographicalId());
  return factors[layer][string] * pred.angle(curvature, r);
}
