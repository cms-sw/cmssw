#ifndef TkSeedingLayers_SeedingLayer_H
#define TkSeedingLayers_SeedingLayer_H

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

class DetLayer;
class TransientTrackingRecHitBuilder;

namespace edm { class Event; class EventSetup; }
namespace ctfseeding {class HitExtractor; }

namespace ctfseeding {

class SeedingLayer {
public:
  enum Side { Barrel = 0, NegEndcap =1,  PosEndcap = 2 }; 
public:
  typedef  std::vector<TransientTrackingRecHit::ConstRecHitPointer> Hits;
  
  SeedingLayer(){}

  SeedingLayer( const std::string & name, int seqNum,
                GeomDetEnumerators::SubDetector subdet,
                Side side,
                int layerId,
                const std::string& hitBuilderName,
                const HitExtractor * hitExtractor,  
                bool usePredefinedErrors = false, float hitErrorRZ = 0., float hitErrorRPhi=0.);

  std::string name() const;
  int seqNum() const;

  void hits(const edm::Event& ev, const edm::EventSetup& es, Hits &) const;
  Hits hits(const edm::Event& ev, const edm::EventSetup& es) const;

  bool operator==(const SeedingLayer &s) const { return name()==s.name(); }

  const DetLayer*  detLayer(const edm::EventSetup& es) const;

  const TransientTrackingRecHitBuilder * hitBuilder(const edm::EventSetup& es) const;

  bool hasPredefinedHitErrors() const;
  float predefinedHitErrorRZ() const;
  float predefinedHitErrorRPhi() const;
 
private:
  class SeedingLayerImpl;
  boost::shared_ptr<SeedingLayerImpl> theImpl;
};

}
#endif
