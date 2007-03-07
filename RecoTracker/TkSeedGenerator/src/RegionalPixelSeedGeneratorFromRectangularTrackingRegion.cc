//
// Package:         RecoTracker/TkSeedGeneratorFromRectangularTrackingRegion
// Class:           RegionalPixelSeedGeneratorFromRectangularTrackingRegion
// 

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGeneratorFromRectangularTrackingRegion.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include <DataFormats/VertexReco/interface/Vertex.h>
#include "DataFormats/Math/interface/Vector3D.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace std;
using namespace reco;

RegionalPixelSeedGeneratorFromRectangularTrackingRegion::RegionalPixelSeedGeneratorFromRectangularTrackingRegion(edm::ParameterSet const& conf) : 
  conf_(conf),combinatorialSeedGenerator(conf)
{
  edm::LogInfo ("RegionalPixelSeedGeneratorFromRectangularTrackingRegion")<<"Enter the RegionalPixelSeedGeneratorFromRectangularTrackingRegion";
  produces<TrajectorySeedCollection>();

  ptmin=conf_.getParameter<double>("ptMin");
  originradius=conf_.getParameter<double>("originRadius");
  halflength=conf_.getParameter<double>("originHalfLength");
  deltaEta = conf_.getParameter<double>("deltaEtaRegion");
  deltaPhi = conf_.getParameter<double>("deltaPhiRegion");
  regSrc = conf_.getParameter<edm::InputTag>("RegionSrc");

}

// Virtual destructor needed.
RegionalPixelSeedGeneratorFromRectangularTrackingRegion::~RegionalPixelSeedGeneratorFromRectangularTrackingRegion() { }  

// Functions that gets called by framework every event
void RegionalPixelSeedGeneratorFromRectangularTrackingRegion::produce(edm::Event& e, const edm::EventSetup& es)
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

  //
  
  //Get the jet direction
  typedef std::vector<RectangularEtaPhiTrackingRegion> RectangularEtaPhiTrackingRegionCollection;   
  edm::Handle<RectangularEtaPhiTrackingRegionCollection> regions;
  e.getByLabel(regSrc, regions);
  
  RectangularEtaPhiTrackingRegionCollection::const_iterator iReg = regions->begin();
  for(;iReg != regions->end();iReg++)
	{
	  
	  RectangularEtaPhiTrackingRegion* region = const_cast<RectangularEtaPhiTrackingRegion*>(&(*iReg));
	  combinatorialSeedGenerator.init(*pixelHits,es);
	  combinatorialSeedGenerator.run(*region,*output,es);
	  // write output to file
	}
      //    }   

    LogDebug("Algorithm Performance")<<" number of seeds = "<< output->size();
    e.put(output);
}
