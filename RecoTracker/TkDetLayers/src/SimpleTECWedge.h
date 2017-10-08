#ifndef TkDetLayers_SimpleTECWedge_h
#define TkDetLayers_SimpleTECWedge_h


#include "TECWedge.h"


/** A concrete implementation for TEC wedge
 *  built out of only one det
 */

#pragma GCC visibility push(hidden)
class SimpleTECWedge final : public TECWedge{
 public:
  SimpleTECWedge(const GeomDet* theDet) __attribute__ ((cold));

  ~SimpleTECWedge() override __attribute__ ((cold));
  
  // GeometricSearchDet interface
  const std::vector<const GeomDet*>& basicComponents() const override {return theDets;}

  const std::vector<const GeometricSearchDet*>& components() const override __attribute__ ((cold));
  
  std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const override __attribute__ ((hot));

  void 
  groupedCompatibleDetsV( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est,
                         std::vector<DetGroup> & result) const override __attribute__ ((hot));

 private:
  const GeomDet* theDet;
  std::vector<const GeomDet*> theDets;

};


#pragma GCC visibility pop
#endif 
