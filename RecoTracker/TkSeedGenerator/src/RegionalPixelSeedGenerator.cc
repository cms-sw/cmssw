//
// Package:         RecoTracker/TkSeedGenerator
// Class:           RegionalPixelSeedGenerator
// 

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGenerator.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include <DataFormats/VertexReco/interface/Vertex.h>
#include "DataFormats/Math/interface/Vector3D.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace std;
using namespace reco;

RegionalPixelSeedGenerator::RegionalPixelSeedGenerator(edm::ParameterSet const& conf) : 
  conf_(conf),combinatorialSeedGenerator(conf)
{
  edm::LogInfo ("RegionalPixelSeedGenerator")<<"Enter the RegionalPixelSeedGenerator";
  produces<TrajectorySeedCollection>();

  ptmin=conf_.getParameter<double>("ptMin");
  originradius=conf_.getParameter<double>("originRadius");
  halflength=conf_.getParameter<double>("originHalfLength");
  vertexSrc=conf_.getParameter<string>("vertexSrc");
  deltaEta = conf_.getParameter<double>("deltaEtaRegion");
  deltaPhi = conf_.getParameter<double>("deltaPhiRegion");
  jetSrc = conf_.getParameter<edm::InputTag>("JetSrc");

}

// Virtual destructor needed.
RegionalPixelSeedGenerator::~RegionalPixelSeedGenerator() { }  

// Functions that gets called by framework every event
void RegionalPixelSeedGenerator::produce(edm::Event& e, const edm::EventSetup& es)
{
  double deltaZVertex =  halflength;
  // get Inputs
  edm::Handle<SiPixelRecHitCollection> pixelHits;

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  edm::Handle<reco::VertexCollection> vertices;
  e.getByLabel(vertexSrc,vertices);
  const reco::VertexCollection vertCollection = *(vertices.product());
  reco::VertexCollection::const_iterator ci = vertCollection.begin();
  if(vertCollection.size() > 0) {
    originz = ci->z();
  }else{
    originz = 0.;
    deltaZVertex = 15.;
  }

  //
  // get the pixel Hits
  //
  std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
  e.getByLabel(hitProducer, pixelHits);
  //  e.getByType(pixelHits);

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());    

  //
  
  //Get the jet direction
  edm::Handle<CaloJetCollection> jets;
  e.getByLabel(jetSrc, jets);

  GlobalVector globalVector(0,0,1);

  //    if(jets->size() > 0)
  //    {
      CaloJetCollection::const_iterator iJet = jets->begin();
      for(;iJet != jets->end();iJet++)
	{
	  
	  GlobalVector jetVector((iJet)->p4().x(),(iJet)->p4().y(),(iJet)->p4().z());
	  globalVector = jetVector;
	  
	  
	  RectangularEtaPhiTrackingRegion* etaphiRegion = new  RectangularEtaPhiTrackingRegion(globalVector,
											       GlobalPoint(0,0,originz), 
											       ptmin,
											       originradius,
											       deltaZVertex,
											       deltaEta,
											       deltaPhi);
	  
	  combinatorialSeedGenerator.init(*pixelHits,es);
	  combinatorialSeedGenerator.run(*etaphiRegion,*output,es);
	  // write output to file
	}
      //    }   

    LogDebug("Algorithm Performance")<<" number of seeds = "<< output->size();
    e.put(output);
}
