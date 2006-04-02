#ifndef CosmicTrackFinder_h
#define CosmicTrackFinder_h

// Package:    RecoTracker/SingleTrackPattern
// Class:      CosmicTrackFinder
// Original Author:  Michele Pioppi-INFN perugia


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"



namespace cms
{
  class CosmicTrackFinder : public edm::EDProducer
  {
  public:

    explicit CosmicTrackFinder(const edm::ParameterSet& conf);

    virtual ~CosmicTrackFinder();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    CosmicTrajectoryBuilder cosmicTrajectoryBuilder_;
    edm::ParameterSet conf_;

  };
}

#endif
