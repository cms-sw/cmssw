#ifndef CtfSpecialSeedGenerator_H
#define CtfSpecialSeedGenerator_H

/** \class CombinatorialSeedGeneratorForCOsmics
 *  A concrete seed generator providing seeds constructed 
 *  from combinations of hits in pairs of strip layers 
 */
//FWK
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
//DataFormats
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"    
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
//RecoTracker
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/SpecialSeedGenerators/interface/SeedFromGenericPairOrTriplet.h"
//#include "RecoTracker/SpecialSeedGenerators/interface/GenericPairOrTripletGenerator.h"
//#include "RecoTracker/SpecialSeedGenerators/interface/SeedCleaner.h"
//MagneticField
#include "MagneticField/Engine/interface/MagneticField.h"
//TrackingTools
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationDirection.h"
//Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/SpecialSeedGenerators/interface/ClusterChecker.h"

#include <map>

class CtfSpecialSeedGenerator : public edm::EDProducer
{
 public:
  typedef TrajectoryStateOnSurface TSOS;
  

  CtfSpecialSeedGenerator(const edm::ParameterSet& conf);

  virtual ~CtfSpecialSeedGenerator();//{};

  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;	
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;	

  virtual void produce(edm::Event& e, const edm::EventSetup& c) override;

 private:
  
  bool run(const edm::EventSetup& c, 
	    const edm::Event& e, 
            TrajectorySeedCollection& output);

  bool buildSeeds(const edm::EventSetup& iSetup,
                  const edm::Event& e,
		  const OrderedSeedingHits& osh,
		  const NavigationDirection& navdir,
                  const PropagationDirection& dir,
                  TrajectorySeedCollection& output);
  //checks that the hits used are at positive y and are on different layers
  bool preliminaryCheck(const SeedingHitSet& shs, const edm::EventSetup& es);
  //We can check if the seed  points in a region covered by scintillators. To be used only in noB case
  //because it uses StraightLinePropagation
  bool postCheck(const TrajectorySeed& seed);
 
 private:
  edm::ParameterSet conf_;
  edm::ESHandle<MagneticField> theMagfield;
  edm::ESHandle<TrackerGeometry> theTracker;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  //edm::ESHandle<SeedCleaner> theCleaner;
  //OrderedHitsGenerator*  hitsGeneratorOutIn;
  //OrderedHitsGenerator*  hitsGeneratorInOut;
  //PropagationDirection inOutPropagationDirection;
  //PropagationDirection outInPropagationDirection;
  //GenericPairOrTripletGenerator* hitsGeneratorOutIn;
  //GenericPairOrTripletGenerator* hitsGeneratorInOut;	
  std::vector<std::unique_ptr<OrderedHitsGenerator> > theGenerators;
  std::vector<PropagationDirection> thePropDirs;
  std::vector<NavigationDirection>  theNavDirs; 
  TrackingRegionProducer* theRegionProducer;	
  //TrajectoryStateTransform theTransformer;
  SeedFromGenericPairOrTriplet* theSeedBuilder; 
  bool useScintillatorsConstraint;
  BoundPlane::BoundPlanePointer upperScintillator;
  BoundPlane::BoundPlanePointer lowerScintillator;
  bool requireBOFF;
  int32_t theMaxSeeds;
  ClusterChecker check;
};
#endif


