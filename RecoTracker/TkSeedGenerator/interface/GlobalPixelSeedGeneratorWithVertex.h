#ifndef GlobalPixelSeedGeneratorWithVertex_h
#define GlobalPixelSeedGeneratorWithVertex_h

//
// Package:         RecoTracker/GlobalPixelSeedGeneratorWithVertex
// Class:           GlobalPixelSeedGeneratorWithVertex
// 
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find TrackingSeeds.


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromPixelWithVertex.h"


class GlobalPixelSeedGeneratorWithVertex : public edm::EDProducer
{
 public:

  explicit GlobalPixelSeedGeneratorWithVertex(const edm::ParameterSet& conf);

  virtual ~GlobalPixelSeedGeneratorWithVertex();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  edm::ParameterSet conf_;
  CombinatorialSeedGeneratorFromPixelWithVertex  combinatorialSeedGenerator;
};

#endif
