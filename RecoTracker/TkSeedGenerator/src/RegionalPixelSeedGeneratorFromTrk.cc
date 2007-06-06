//
// Package:         RecoTracker/TkSeedGeneratorFromTrk
// Class:           RegionalPixelSeedGeneratorFromTrk
//
 
#include <iostream>
#include <memory>
#include <string>
 
#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGeneratorFromTrk.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/Vector3D.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
 
using namespace std;
using namespace reco;
 
RegionalPixelSeedGeneratorFromTrk::RegionalPixelSeedGeneratorFromTrk(edm::ParameterSet const& conf) :
  conf_(conf),combinatorialSeedGenerator(conf)
{
  edm::LogInfo ("RegionalPixelSeedGeneratorFromTrk")<<"Enter the  RegionalPixelSeedGeneratorFromTrk";
  produces<TrajectorySeedCollection>();
 
  ptmin=conf_.getParameter<double>("ptMin");
  vertexZconstrained = conf_.getParameter<bool>("vertexZConstrained");
  vertexzDefault=conf_.getParameter<double>("vertexZDefault");
  vertexSrc=conf_.getParameter<string>("vertexSrc");
  originradius=conf_.getParameter<double>("originRadius");
  halflength=conf_.getParameter<double>("originHalfLength");
  deltaEta = conf_.getParameter<double>("deltaEtaRegion");
  deltaPhi = conf_.getParameter<double>("deltaPhiRegion");
  trkSrc = conf_.getParameter<edm::InputTag>("TrkSrc");
 
}
 
// Virtual destructor needed.
RegionalPixelSeedGeneratorFromTrk::~RegionalPixelSeedGeneratorFromTrk( ) { }
 
// Functions that gets called by framework every event
void RegionalPixelSeedGeneratorFromTrk::produce(edm::Event& e,  const edm::EventSetup& es)
{
  // get Inputs
  edm::Handle<SiPixelRecHitCollection> pixelHits;
 
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
 
  //
  // get the pixel Hits
  //
  std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
  e.getByLabel(hitProducer, pixelHits);
  //  e.getByType(pixelHits);
 
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
 
  // get highest Pt pixel vertex (if existing)
  double deltaZVertex =  halflength;
  if (vertexSrc.length()>1) {
      edm::Handle<reco::VertexCollection> vertices;
      e.getByLabel(vertexSrc,vertices);
      const reco::VertexCollection vertCollection = *(vertices.product());
      reco::VertexCollection::const_iterator ci = vertCollection.begin();
      if (vertCollection.size()>0) {
            originz = ci->z();
      } else {
            originz = vertexzDefault;
            deltaZVertex = 15.;
      }
  }
 
  //Get the track directions
  edm::Handle<TrackCollection> trks;
  e.getByLabel(trkSrc, trks);
 
     TrackCollection::const_iterator iTrk = trks->begin();
      for(;iTrk != trks->end();iTrk++)
	{
	
        double vz = originz;
        // Change Z vertex position according to track parameters  if requested
        if (vertexZconstrained) {
            vz = iTrk->dz();
        }
	
	  GlobalVector dirVector((iTrk)->px(),(iTrk)->py(),(iTrk)->pz());
	
	  RectangularEtaPhiTrackingRegion etaphiRegion(dirVector,
              GlobalPoint(0,0,float(vz)), ptmin, originradius, deltaZVertex, deltaEta, deltaPhi);
	
	  combinatorialSeedGenerator.init(*pixelHits,es);
	  combinatorialSeedGenerator.run(etaphiRegion,*output,es);
	}
 
    LogDebug("Algorithm Performance")<<" number of seeds = "<<  output->size();
    e.put(output);
}
