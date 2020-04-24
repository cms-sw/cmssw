#ifndef TkDetLayers_Phase2OTtiltedBarrelLayer_h
#define TkDetLayers_Phase2OTtiltedBarrelLayer_h


#include "TrackingTools/DetLayers/interface/RodBarrelLayer.h"
#include "Phase2OTBarrelLayer.h"
#include "Phase2OTBarrelRod.h"
#include "Phase2EndcapRing.h"
#include "SubLayerCrossings.h"


/** A concrete implementation for Phase2OTtiltedBarrel layer 
 *  built out of BarrelPhase2OTBarrelRod
 */

#pragma GCC visibility push(hidden)
class Phase2OTtiltedBarrelLayer final : public Phase2OTBarrelLayer {
 public:

  Phase2OTtiltedBarrelLayer(std::vector<const Phase2OTBarrelRod*>& innerRods,
                            std::vector<const Phase2OTBarrelRod*>& outerRods,
                            std::vector<const Phase2EndcapRing*>& negRings,
                            std::vector<const Phase2EndcapRing*>& posRings);
  
  ~Phase2OTtiltedBarrelLayer() override;
  
  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const final;
    
 private:
  std::vector<const GeometricSearchDet*> theNegativeRingsComps;
  std::vector<const GeometricSearchDet*> thePositiveRingsComps;

  ReferenceCountingPointer<BoundCylinder>  theCylinder;
  
};


#pragma GCC visibility pop
#endif 
