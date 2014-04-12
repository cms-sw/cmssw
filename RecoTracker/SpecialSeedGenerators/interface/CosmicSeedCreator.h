#ifndef RecoTracker_TkSeedGenerator_CosmicSeedCreator_H
#define RecoTracker_TkSeedGenerator_CosmicSeedCreator_H

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"



class CosmicSeedCreator : public SeedCreator {

 public:
  CosmicSeedCreator( const edm::ParameterSet & extra )
    {
      maxseeds_ = extra.getParameter<int>("maxseeds");
    }

  virtual ~CosmicSeedCreator(){}

  // initialize the "event dependent state"
  virtual void init(const TrackingRegion & region,
		    const edm::EventSetup& es,
		    const SeedComparitor *filter);

  // make job 
  // fill seedCollection with the "TrajectorySeed"
  virtual void makeSeed(TrajectorySeedCollection & seedCollection,
			const SeedingHitSet & hits);

  
private:
  const TrackingRegion * region = nullptr;
  const SeedComparitor *filter = nullptr;
  edm::ESHandle<MagneticField> bfield;

  unsigned int maxseeds_;
};
#endif 
