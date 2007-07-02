#ifndef CombinatorialSeedGeneratorForCosmics_H
#define CombinatorialSeedGeneratorForCosmics_H

/** \class CombinatorialSeedGeneratorForCOsmics
 *  A concrete seed generator providing seeds constructed 
 *  from combinations of hits in pairs of strip layers 
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"    
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromLayerPairs.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcherESProducer.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <map>

class CombinatorialSeedGeneratorForCosmics : public edm::EDProducer
{
 public:
  typedef TrajectoryStateOnSurface TSOS;
  

  CombinatorialSeedGeneratorForCosmics(const edm::ParameterSet& conf);

  virtual ~CombinatorialSeedGeneratorForCosmics();//{};

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  void init(const SiStripRecHit2DCollection &collstereo,
	    const SiStripRecHit2DCollection &collrphi,
	    const SiStripMatchedRecHit2DCollection &collmatched,
	    const edm::EventSetup& c);

  void  run(TrajectorySeedCollection &,const edm::EventSetup& c);

  void  seeds(TrajectorySeedCollection &output,
	      const edm::EventSetup& c);
 
 private:
  bool isFirstCall;
  edm::ParameterSet conf_;
  edm::ESHandle<MagneticField> magfield;
  edm::ESHandle<TrackerGeometry> tracker;
  edm::ESHandle<SiStripRecHitMatcher> rechitmatcher;
  TrajectoryStateTransform transformer;
  std::string geometry;
  OrderedHitPairs HitPairs;
  const SiStripRecHit2DCollection* stereocollection;
  float p;
  float meanRadius;
  bool useScintillatorsConstraint;
  BoundPlane* upperScintillator;
  BoundPlane* lowerScintillator;
  bool checkDirection(const FreeTrajectoryState& state,
                      const MagneticField* magField);	
  const TrajectorySeed* buildSeed(const TrackingRecHit* first,
                 const TrackingRecHit* second,
                 const PropagationDirection& dir); //,
                 //edm::OwnVector<TrajectorySeed*>& outseed);	
  edm::OwnVector<TrackingRecHit> match(const TrackingRecHit* hit, 
             const GlobalVector& direction);//,
             //edm::OwnVector<TrackingRecHit>& hits);
  std::pair<GlobalPoint, GlobalError> toGlobal(const TrackingRecHit* rechit);
};
#endif


