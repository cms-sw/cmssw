#ifndef RecoLocalTracker_SiPhase2Clusterizer_hits_H
#define RecoLocalTracker_SiPhase2Clusterizer_hits_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "SimTracker/SiPhase2Digitizer/interface/PixelClusterSimLink.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

#include <vector>
#include <map>

typedef std::map< unsigned int, std::vector< struct ValHit > > ValHitsCollection;
typedef std::vector< struct ValHit > ValHitsVector;
typedef struct ValHit ValHit;

struct ValHit {
    double x, y;
    LocalPoint localPos;
    GlobalPoint globalPos;
    double xx, xy, yy;
    std::vector< unsigned int > simTracks;
    edm::Ref< edmNew::DetSetVector< SiPixelCluster >, SiPixelCluster > cluster;
};



ValHitsCollection ValHitsBuilder(const TrackerGeometry* tkGeom, edm::DetSetVector< PixelClusterSimLink >* clusterLinks);
ValHitsCollection ValHitsBuilder(const TrackerGeometry* tkGeom, edm::DetSetVector< PixelClusterSimLink >* clusterLinks, edmNew::DetSetVector< SiPixelRecHit >* recHits);

#endif
