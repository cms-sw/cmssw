//
// Package:         RecoTracker/TkSeedGenerator
// Class:           GlobalPixelSeedGeneratorWithVertex
// 

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/TkSeedGenerator/interface/GlobalPixelSeedGeneratorWithVertex.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

using namespace std;
GlobalPixelSeedGeneratorWithVertex::GlobalPixelSeedGeneratorWithVertex(edm::ParameterSet const& conf) : 
  conf_(conf),combinatorialSeedGenerator(conf)
{
  edm::LogInfo ("GlobalPixelSeedGeneratorWithVertex")<<"Enter the GlobalPixelSeedGeneratorWithVertex";
  produces<TrajectorySeedCollection>();
}

// Virtual destructor needed.
GlobalPixelSeedGeneratorWithVertex::~GlobalPixelSeedGeneratorWithVertex() { }  

// Functions that gets called by framework every event
void GlobalPixelSeedGeneratorWithVertex::produce(edm::Event& e, const edm::EventSetup& es)
{
  // get Inputs
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  edm::Handle<reco::VertexCollection> pixelVertices;

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  //
  // get the pixel Hits
  //
  std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
  e.getByLabel(hitProducer, pixelHits);
  std::string vertexProducer = conf_.getParameter<std::string>("VertexProducer");
  e.getByLabel(vertexProducer, pixelVertices);
  //  e.getByType(pixelHits);

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);
  //

  // should they be put together in an unique method 
  combinatorialSeedGenerator.init(*pixelHits, *pixelVertices, es);
  combinatorialSeedGenerator.run(*output,es);

  // write output to file
  LogDebug("Algorithm Performance")<<" number of seeds = "<< output->size();


  e.put(output);
}
