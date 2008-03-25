#include "RecoTracker/TkDetLayers/interface/SimpleTECWedge.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkDetLayers/interface/ForwardDiskSectorBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

SimpleTECWedge::SimpleTECWedge(const GeomDet* theInputDet):
  theDet(theInputDet)
{
  theDets.push_back(theDet);

  theDiskSector = ForwardDiskSectorBuilderFromDet()( theDets );
  
  LogDebug("TkDetLayers") << "DEBUG INFO for CompositeTECWedge" << "\n"
			  << "TECWedge z, perp,innerRadius,outerR: " 
			  << this->position().z() << " , "
			  << this->position().perp() << " , "
			  << theDiskSector->innerRadius() << " , "
			  << theDiskSector->outerRadius() ;

}

SimpleTECWedge::~SimpleTECWedge(){

} 


const vector<const GeometricSearchDet*>& 
SimpleTECWedge::components() const{
  throw DetLayerException("SimpleTECWedge doesn't have GeometricSearchDet components");
}

  
pair<bool, TrajectoryStateOnSurface>
SimpleTECWedge::compatible( const TrajectoryStateOnSurface& tsos,
			    const Propagator& prop, 
			    const MeasurementEstimator& est) const
{
  GeomDetCompatibilityChecker theCompatibilityChecker;
  pair<bool, TrajectoryStateOnSurface> 
    compat = theCompatibilityChecker.isCompatible( theDet,tsos, prop, est);
  
  return compat;
}



vector<DetGroup> 
SimpleTECWedge::groupedCompatibleDets( const TrajectoryStateOnSurface& tsos,
				       const Propagator& prop,
				       const MeasurementEstimator& est) const
{
  pair<bool, TrajectoryStateOnSurface> compat = this->compatible(tsos,prop,est);

  vector<DetGroup> result;
  if (compat.first) {
    result.push_back( DetGroup(0,1) ); 
    DetGroupElement ge( theDet, compat.second);
    result.front().push_back(ge);
  }

  return result;
}



