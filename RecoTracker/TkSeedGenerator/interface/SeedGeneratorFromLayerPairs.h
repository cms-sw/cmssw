#ifndef SeedGeneratorFromLayerPairs_H
#define SeedGeneratorFromLayerPairs_H

/** \class SeedGeneratorFromLayerPairs
 *  A concrete regional seed generator providing seeds constructed from
 *  combinations of hits in provided (at construction) layer pairs 
 */ 

#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromHitPairsConsecutiveHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "FWCore/Framework/interface/EventSetup.h"
//class TrackingRegionFactory;
class SeedLayerPairs;

class SeedGeneratorFromLayerPairs
          : public SeedGeneratorFromHitPairsConsecutiveHits {
public:

  /// dummy constructor
  SeedGeneratorFromLayerPairs(const edm::ParameterSet& conf)
    :     SeedGeneratorFromHitPairsConsecutiveHits(conf), 
    thePairGenerator(0), 
    theRegion(0)
    { }

  /** Construct from layerPairs only. 
   *  In order to obtain seed one must call
   *  seeds(const TrackingRegion &region) rather than seeds()
   */
	    //   SeedGeneratorFromLayerPairs( const SeedLayerPairs * layerPairs);
    SeedGeneratorFromLayerPairs( SeedLayerPairs * layerPairs, const edm::ParameterSet& conf);
  /** Construct from a TrackingRegionFactory.
   *  The TrackingRegionFactory is expected to be valid throughout
   *  the lifetime of this object. If the factory
   *  is deleted or relocated and one of the methods requiring regions
   *  is called, the thing will crash.
   */
/*   SeedGeneratorFromLayerPairs(  */
/*       const SeedLayerPairs * layerPairs, */
/*       const TrackingRegionFactory& regionFactory); */

  /** Construct with an existing GlobalTrackingRegion.
   *  The region is cloned, and an independed copy is used.
   */
  SeedGeneratorFromLayerPairs( 
      const SeedLayerPairs * layerPairs,
      const TrackingRegion& region);

  /** Construct from minimal P_t, and origin region size and position.
   *  A GlobalTrackingRegion is constructed from this information
   *  and used internally.
   */
  SeedGeneratorFromLayerPairs(
      const SeedLayerPairs * layerPairs,
      float ptMin, float originRadius, float originHalfLength,float originZPos);

  /// destructor
  virtual ~SeedGeneratorFromLayerPairs(){};

  virtual const TrackingRegion * trackingRegion() const;

protected:
  //  virtual void initPairGenerator( const SeedLayerPairs * layerPairs);
  virtual void initPairGenerator( SeedLayerPairs * layerPairs,const edm::EventSetup& iSetup);
  virtual HitPairGenerator * pairGenerator() const;

private:
  std::auto_ptr<HitPairGenerator> thePairGenerator;
  //  const TrackingRegionFactory * theRegionFactory;
  DeepCopyPointerByClone<TrackingRegion>  theRegion;

};
#endif

