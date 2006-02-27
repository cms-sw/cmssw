#include "RecoTracker/TkDetLayers/interface/PixelRod.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

typedef GeometricSearchDet::DetWithState DetWithState;

PixelRod::PixelRod(vector<const GeomDet*>& theInputDets):
  DetRodOneR(theInputDets.begin(),theInputDets.end())
{
  theBinFinder = BinFinderType(basicComponents().begin(), basicComponents().end());
}

PixelRod::~PixelRod(){

} 

vector<const GeometricSearchDet*> 
PixelRod::components() const{
  cout << "temporary dummy implementation of PixelRod::components()!!" << endl;
  return vector<const GeometricSearchDet*>();
}
 
 
pair<bool, TrajectoryStateOnSurface>
PixelRod::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		      const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of PixelRod::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
PixelRod::compatibleDets( const TrajectoryStateOnSurface& startingState,
			  const Propagator& prop, 
			  const MeasurementEstimator& est) const
{  
  typedef MeasurementEstimator::Local2DVector Local2DVector;
  TrajectoryStateOnSurface ts = prop.propagate( startingState, specificSurface());
  if (!ts.isValid()) return vector<DetWithState>();  

  GlobalPoint startPos = ts.globalPosition();

  const vector<const GeomDet*> theDets = basicComponents();
  vector<DetWithState> result;
  
  int closest = theBinFinder.binIndex(startPos.z());
  pair<bool,TrajectoryStateOnSurface> closestCompat = 
    theCompatibilityChecker.isCompatible(theDets[closest],startingState, prop, est);
  
  if ( closestCompat.first) {
    result.push_back( DetWithState( theDets[closest], closestCompat.second));
  }

  const BoundPlane& closestPlane( dynamic_cast<const BoundPlane&>( 
				  theDets[closest]->surface()));
  
  Local2DVector maxDistance = 
    est.maximalLocalDisplacement( closestCompat.second, closestPlane);
  
  float detHalfLen = theDets[closest]->surface().bounds().length()/2.;
  
  // explore neighbours

  for (size_t idet=closest+1; idet < theDets.size(); idet++) {
    LocalPoint nextPos( theDets[idet]->surface().toLocal( closestCompat.second.globalPosition()));
    if (fabs(nextPos.y()) < detHalfLen + maxDistance.y()) {
      if ( !add(idet, result, startingState, prop, est)) break;
    } else {
      break;
    }
  }
  for (int idet=closest-1; idet >= 0; idet--) {
    LocalPoint nextPos( theDets[idet]->surface().toLocal( closestCompat.second.globalPosition()));
    if (fabs(nextPos.y()) < detHalfLen + maxDistance.y()) {
      if ( !add(idet, result, startingState, prop, est)) break;
    } else {
      break;
    }
  }

  return result;
}


vector<DetGroup> 
PixelRod::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop,
				 const MeasurementEstimator& est) const
{
  cout << "dummy implementation of PixelRod::groupedCompatibleDets()" << endl;
  return vector<DetGroup>();
}



