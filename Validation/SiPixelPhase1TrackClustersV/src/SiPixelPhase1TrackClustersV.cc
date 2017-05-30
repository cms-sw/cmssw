// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackClustersV
// Class:       SiPixelPhase1TrackClustersV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "Validation/SiPixelPhase1TrackClustersV/interface/SiPixelPhase1TrackClustersV.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"


SiPixelPhase1TrackClustersV::SiPixelPhase1TrackClustersV(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig) 
{
  clustersToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("clusters"));
  tracksToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
}

void SiPixelPhase1TrackClustersV::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // get geometry
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  assert(tracker.isValid());
  
  //get the map
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken( tracksToken_, tracks);
  
  // get clusters
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> >  clusterColl;
  iEvent.getByToken( clustersToken_, clusterColl );
  
  // we need to store some per-cluster data. Instead of a map, we use a vector,
  // exploiting the fact that all custers live in the DetSetVector and we can 
  // use the same indices to refer to them.
  // corr_charge is not strictly needed but cleaner to have it.
  std::vector<bool>  ontrack    (clusterColl->data().size(), false);
  std::vector<float> corr_charge(clusterColl->data().size(), -1.0f);

  for (auto const & track : *tracks) {

    auto const & trajParams = track.extra()->trajParams();
    assert(trajParams.size()==track.recHitsSize());
    auto hb = track.recHitsBegin();
    for(unsigned int h=0;h<track.recHitsSize();h++){
      auto hit = *(hb+h);
      if (!hit->isValid()) continue;
      DetId id = hit->geographicalId();

      // check that we are in the pixel
      uint32_t subdetid = (id.subdetId());
      if (subdetid != PixelSubdetector::PixelBarrel && subdetid != PixelSubdetector::PixelEndcap) continue;
      auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
      if (!pixhit) continue;

      // get the cluster
      auto clust = pixhit->cluster();
      if (clust.isNull()) continue;
      ontrack[clust.key()] = true; // mark cluster as ontrack


      // correct charge for track impact angle
      auto const & ltp = trajParams[h];
      LocalVector localDir = ltp.momentum()/ltp.momentum().mag();

      float clust_alpha = atan2(localDir.z(), localDir.x());
      float clust_beta  = atan2(localDir.z(), localDir.y());
      double corrCharge = clust->charge()/1000. * sqrt( 1.0 / ( 1.0/pow( tan(clust_alpha), 2 ) + 
                                                          1.0/pow( tan(clust_beta ), 2 ) + 
                                                          1.0 ));
      corr_charge[clust.key()] = (float) corrCharge;
    }

  edmNew::DetSetVector<SiPixelCluster>::const_iterator it;
  for (it = clusterColl->begin(); it != clusterColl->end(); ++it) {
    auto id = DetId(it->detId());

    for(auto subit = it->begin(); subit != it->end(); ++subit) {
      // we could do subit-...->data().front() as well, but this seems cleaner.
      auto key = edmNew::makeRefTo(clusterColl, subit).key(); 
      float corrected_charge = corr_charge[key];
      SiPixelCluster const& cluster = *subit;

      histo[CHARGE].fill(double(corrected_charge), id, &iEvent);
      histo[SIZE_X].fill(double(cluster.sizeX() ), id, &iEvent);
      histo[SIZE_Y].fill(double(cluster.sizeY() ), id, &iEvent);
      }
    }
  }

}

DEFINE_FWK_MODULE(SiPixelPhase1TrackClustersV);

