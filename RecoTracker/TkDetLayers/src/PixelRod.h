#ifndef TkDetLayers_PixelRod_h
#define TkDetLayers_PixelRod_h


#include "TrackingTools/DetLayers/interface/DetRodOneR.h"
#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInZ.h"

/** A concrete implementation for PixelRod
 */

#pragma GCC visibility push(hidden)
class PixelRod final : public DetRodOneR{
 public:
    typedef PeriodicBinFinderInZ<float>   BinFinderType;

  PixelRod(std::vector<const GeomDet*>& theDets);
  ~PixelRod() override;
  
  // GeometricSearchDet interface

  const std::vector<const GeometricSearchDet*>& components() const override __attribute__ ((cold));
  
  std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const override;

  void
  compatibleDetsV( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est,
		  std::vector<DetWithState> & result) const override __attribute__ ((hot));

  void  
  groupedCompatibleDetsV( const TrajectoryStateOnSurface&,
			 const Propagator&,
			 const MeasurementEstimator&,
			 std::vector<DetGroup> &) const override;


 private:
  BinFinderType theBinFinder;
      
  
};


#pragma GCC visibility pop
#endif 
