#ifndef TkDetLayers_SimpleTECWedge_h
#define TkDetLayers_SimpleTECWedge_h


#include "TECWedge.h"


/** A concrete implementation for TEC wedge
 *  built out of only one det
 */

#pragma GCC visibility push(hidden)
class SimpleTECWedge GCC11_FINAL : public TECWedge{
 public:
  SimpleTECWedge(const GeomDet* theDet);

  ~SimpleTECWedge();
  
  // GeometricSearchDet interface
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theDets;}

  virtual const std::vector<const GeometricSearchDet*>& components() const;
  
  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  virtual void 
  groupedCompatibleDetsV( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est,
                         std::vector<DetGroup> & result) const;

 private:
  const GeomDet* theDet;
  std::vector<const GeomDet*> theDets;

};


#pragma GCC visibility pop
#endif 
