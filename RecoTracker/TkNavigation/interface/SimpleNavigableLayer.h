#ifndef TkNavigation_SimpleNavigableLayer_H
#define TkNavigation_SimpleNavigableLayer_H

#include "TrackingTools/DetLayers/interface/NavigableLayer.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include <vector>

class GtfPropagator;

/** A partial implementation of the NavigableLayer
 */

class SimpleNavigableLayer : public NavigableLayer {
public:

  typedef vector<const DetLayer*>              DLC;
  typedef vector<BarrelDetLayer*>              BDLC;
  typedef vector<ForwardDetLayer*>             FDLC;

  SimpleNavigableLayer( const MagneticField* field,float eps) :
    theEpsilon(eps),thePropagator(field) {}

  virtual void setInwardLinks(const BDLC&, const FDLC&) = 0;

protected:

  typedef BDLC::iterator                       BDLI;
  typedef FDLC::iterator                       FDLI;
  typedef BDLC::const_iterator                 ConstBDLI;
  typedef FDLC::const_iterator                 ConstFDLI;
  typedef TrajectoryStateOnSurface             TSOS;

  float                     theEpsilon;

  mutable AnalyticalPropagator     thePropagator;

  bool wellInside( const FreeTrajectoryState& fts, PropagationDirection dir,
		   const BarrelDetLayer* bl, DLC& result) const;

  bool wellInside( const FreeTrajectoryState& fts, PropagationDirection dir,
		   const ForwardDetLayer* bl, DLC& result) const;

  bool wellInside( const FreeTrajectoryState& fts, PropagationDirection dir,
		   ConstBDLI begin, ConstBDLI end, DLC& result) const;

  bool wellInside( const FreeTrajectoryState& fts, PropagationDirection dir,
		   const DLC& layers, DLC& result) const;

  bool wellInside( const FreeTrajectoryState& fts, PropagationDirection dir,
		   ConstFDLI begin, ConstFDLI end, DLC& result) const;

  Propagator& propagator( PropagationDirection dir) const;

  void pushResult( DLC& result, const FDLC& tmp) const;
  void pushResult( DLC& result, const BDLC& tmp) const;

};

#endif // SimpleNavigableLayer_H
