#ifndef CombinedHitPairGenerator_H
#define CombinedHitPairGenerator_H

#include <vector>
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackingRegion;
class OrderedHitPairs;
class HitPairGeneratorFromLayerPair;
namespace ctfseeding { class SeedingLayer;}
namespace edm { class Event; class EventSetup; }

#include "FWCore/Framework/interface/ESWatcher.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"

/** \class CombinedHitPairGenerator
 * Hides set of HitPairGeneratorFromLayerPair generators.
 */

class CombinedHitPairGenerator : public HitPairGenerator {
public:
  typedef LayerHitMapCache LayerCacheType;

public:
  CombinedHitPairGenerator(const edm::ParameterSet & cfg);
  virtual ~CombinedHitPairGenerator();

  void  add(const ctfseeding::SeedingLayer & inner, 
	      const ctfseeding::SeedingLayer & outer);

  /// form base class
  virtual void hitPairs( const TrackingRegion& reg, 
      OrderedHitPairs & result, const edm::Event& ev, const edm::EventSetup& es);

  /// from base class
  virtual CombinedHitPairGenerator * clone() const 
    { return new CombinedHitPairGenerator(theConfig); } 

private:
  CombinedHitPairGenerator(const CombinedHitPairGenerator & cb); 
  void init(const ctfseeding::SeedingLayerSets & layerSets);
  void init(const edm::ParameterSet & cfg, const edm::EventSetup& es);
  void cleanup();


  mutable bool initialised;
  edm::ParameterSet theConfig;

  LayerCacheType   theLayerCache;

  edm::ESWatcher<TrackerDigiGeometryRecord> theESWatcher;

  typedef std::vector<HitPairGeneratorFromLayerPair *>   Container;
  Container        theGenerators;

};
#endif
