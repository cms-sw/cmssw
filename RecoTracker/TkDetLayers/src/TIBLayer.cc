#include "TIBLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"

#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"
#include "TrackingTools/DetLayers/src/DetLessZ.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

TIBLayer::TIBLayer(vector<const TIBRing*>& innerRings,
		   vector<const TIBRing*>& outerRings) : 
  theInnerComps(innerRings.begin(),innerRings.end()), 
  theOuterComps(outerRings.begin(),outerRings.end())
{
  theComps.assign(theInnerComps.begin(),theInnerComps.end());
  theComps.insert(theComps.end(),theOuterComps.begin(),theOuterComps.end());
  
  sort(theComps.begin(),theComps.end(),DetLessZ());
  sort(theInnerComps.begin(),theInnerComps.end(),DetLessZ());
  sort(theOuterComps.begin(),theOuterComps.end(),DetLessZ());
  
  for(vector<const GeometricSearchDet*>::const_iterator it=theComps.begin();
      it!=theComps.end();it++){  
    theBasicComps.insert(theBasicComps.end(),	
			 (**it).basicComponents().begin(),
			 (**it).basicComponents().end());
  }

  // initialize the surface
  theInnerCylinder = cylinder( theInnerComps);
  theOuterCylinder = cylinder( theOuterComps);
  initialize();
  
  LogDebug("TkDetLayers") << "==== DEBUG TIBLayer =====" ; 
  LogDebug("TkDetLayers") << "innerCyl radius, thickness, lenght: " 
			  << theInnerCylinder->radius() << " , "
			  << theInnerCylinder->bounds().thickness() << " , "
			  << theInnerCylinder->bounds().length() ;
 
  LogDebug("TkDetLayers") << "outerCyl radius, thickness, lenght: " 
			  << theOuterCylinder->radius() << " , "
			  << theOuterCylinder->bounds().thickness() << " , "
			  << theOuterCylinder->bounds().length() ;

  LogDebug("TkDetLayers") << "Cyl radius, thickness, lenght: " 
			  << specificSurface().radius() << " , "
			  << specificSurface().bounds().thickness() << " , "
			  << specificSurface().bounds().length() ;

  for (vector<const GeometricSearchDet*>::const_iterator i=theInnerComps.begin();
       i != theInnerComps.end(); i++){
    LogDebug("TkDetLayers") << "inner TIBRing pos z,radius,eta,phi: " 
			    << (**i).position().z() << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta() << " , " 
			    << (**i).position().phi() ;
  }

  for (vector<const GeometricSearchDet*>::const_iterator i=theOuterComps.begin();
       i != theOuterComps.end(); i++){
    LogDebug("TkDetLayers") << "outer TIBRing pos z,radius,eta,phi: " 
			    << (**i).position().z() << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta() << " , " 
			    << (**i).position().phi() ;
  }
  


  // initialise the bin finders
  //  vector<const GeometricSearchDet*> tmpIn;
  //for (vector<const TIBRing*>::const_iterator i=theInnerRings.begin();
  //     i != theInnerRings.end(); i++) tmpIn.push_back(*i);
  theInnerBinFinder = GeneralBinFinderInZforGeometricSearchDet<float>(theInnerComps.begin(), 
								      theInnerComps.end());
 
  theOuterBinFinder = GeneralBinFinderInZforGeometricSearchDet<float>(theOuterComps.begin(),
								      theOuterComps.end());
}

TIBLayer::~TIBLayer(){
  vector<const GeometricSearchDet*>::const_iterator i;
  for (i=theComps.begin(); i!=theComps.end(); i++) {
    delete *i;
  }
} 

  


BoundCylinder* 
TIBLayer::cylinder( const vector<const GeometricSearchDet*>& rings)
{
  float leftPos = rings.front()->surface().position().z();
  float rightPos = rings.back()->surface().position().z();

  const BoundCylinder & frontRing = static_cast<const BoundCylinder &>(rings.front()->surface());
  const BoundCylinder & backRing  = static_cast<const BoundCylinder &>(rings.back()->surface());
  float r = frontRing.radius(); 
  const Bounds& leftBounds  = frontRing.bounds();
  const Bounds& rightBounds = backRing.bounds();

  //float r = rings.front()->specificSurface().radius();
  //const Bounds& leftBounds = rings.front()->specificSurface().bounds();
  //const Bounds& rightBounds = rings.back()->specificSurface().bounds();

  float thick = leftBounds.thickness() / 2;
  float zmin = leftPos  - leftBounds.length() / 2;
  float zmax = rightPos + rightBounds.length() / 2;
  float rmin = r-thick;
  float rmax = r+thick;
  float zpos = 0.5*(leftPos+rightPos);

  auto scp = new SimpleCylinderBounds(rmin, rmax, zmin-zpos, zmax-zpos);
  return new Cylinder(r, Surface::PositionType( 0, 0, zpos),
                         rings.front()->surface().rotation(), scp);

}



void 
TIBLayer::groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
				  const Propagator& prop,
				  const MeasurementEstimator& est,
				  std::vector<DetGroup> & result) const {
  SubLayerCrossings  crossings; 
  crossings = computeCrossings( tsos, prop.propagationDirection());
  if(! crossings.isValid()) return;
  
  vector<DetGroup> closestResult;
  addClosest( tsos, prop, est, crossings.closest(), closestResult);
  // this differs from compatibleDets logic, which checks next in such cases!!!
  if (closestResult.empty())    return;
  
  
  DetGroupElement closestGel( closestResult.front().front());
  float window = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);

  searchNeighbors( tsos, prop, est, crossings.closest(), window,
		   closestResult, false);

  vector<DetGroup> nextResult;
  searchNeighbors( tsos, prop, est, crossings.other(), window,
		   nextResult, true);

  int crossingSide = LayerCrossingSide().barrelSide( closestGel.trajectoryState(), prop);
  DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult), std::move(nextResult), result, 
					  crossings.closestIndex(), crossingSide);
}

SubLayerCrossings TIBLayer::computeCrossings( const TrajectoryStateOnSurface& startingState,
						      PropagationDirection propDir) const
{
  GlobalPoint startPos( startingState.globalPosition());
  GlobalVector startDir( startingState.globalMomentum());
  double rho( startingState.transverseCurvature());

  HelixBarrelCylinderCrossing innerCrossing( startPos, startDir, rho,
					     propDir,*theInnerCylinder);
  if (!innerCrossing.hasSolution()) return SubLayerCrossings(); 

  GlobalPoint gInnerPoint( innerCrossing.position());
  int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.z());
  const GeometricSearchDet* innerRing( theInnerComps[innerIndex]);
  float innerDist = fabs( innerRing->surface().position().z() - gInnerPoint.z());
  SubLayerCrossing innerSLC( 0, innerIndex, gInnerPoint);

  HelixBarrelCylinderCrossing outerCrossing( startPos, startDir, rho,
					     propDir,*theOuterCylinder);
  if (!outerCrossing.hasSolution()) return SubLayerCrossings();

  GlobalPoint gOuterPoint( outerCrossing.position());
  int outerIndex = theOuterBinFinder.binIndex(gOuterPoint.z());
  const GeometricSearchDet* outerRing( theOuterComps[outerIndex]);
  float outerDist = fabs( outerRing->surface().position().z() - gOuterPoint.z());
  SubLayerCrossing outerSLC( 1, outerIndex, gOuterPoint);

  if (innerDist < outerDist) {
    return SubLayerCrossings( innerSLC, outerSLC, 0);
  }
  else {
    return SubLayerCrossings( outerSLC, innerSLC, 1);
  } 
}

bool TIBLayer::addClosest( const TrajectoryStateOnSurface& tsos,
				      const Propagator& prop,
				      const MeasurementEstimator& est,
				      const SubLayerCrossing& crossing,
				      vector<DetGroup>& result) const
{
//   edm::LogInfo(TkDetLayers) << "Entering TIBLayer::addClosest" ;

  const vector<const GeometricSearchDet*>& sub( subLayer( crossing.subLayerIndex()));
  const Det* det(sub[crossing.closestDetIndex()]);
  return CompatibleDetToGroupAdder().add( *det, tsos, prop, est, result);
}

void TIBLayer::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				const Propagator& prop,
				const MeasurementEstimator& est,
				const SubLayerCrossing& crossing,
				float window, 
				vector<DetGroup>& result,
				bool checkClosest) const
{
  GlobalPoint gCrossingPos = crossing.position();

  const vector<const GeometricSearchDet*>& sLayer( subLayer( crossing.subLayerIndex()));
 
  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex-1;
  int posStartIndex = closestIndex+1;

  if (checkClosest) { // must decide if the closest is on the neg or pos side
    if (gCrossingPos.z() < sLayer[closestIndex]->surface().position().z()) {
      posStartIndex = closestIndex;
    }
    else {
      negStartIndex = closestIndex;
    }
  }

  typedef CompatibleDetToGroupAdder Adder;
  for (int idet=negStartIndex; idet >= 0; idet--) {
    const GeometricSearchDet* neighborRing = sLayer[idet];
    if (!overlap( gCrossingPos, *neighborRing, window)) break;
    if (!Adder::add( *neighborRing, tsos, prop, est, result)) break;
  }
  for (int idet=posStartIndex; idet < static_cast<int>(sLayer.size()); idet++) {
    const GeometricSearchDet* neighborRing = sLayer[idet];
    if (!overlap( gCrossingPos, *neighborRing, window)) break;
    if (!Adder::add( *neighborRing, tsos, prop, est, result)) break;
  }
}

bool TIBLayer::overlap( const GlobalPoint& crossPoint,
			const GeometricSearchDet& det, 
			float window) const
{
  float halfLength = det.surface().bounds().length()/2.;

//   edm::LogInfo(TkDetLayers) << " TIBLayer: checking ring with z " << det.position().z();

  if ( fabs( crossPoint.z()-det.position().z()) < (halfLength + window)) {
//     edm::LogInfo(TkDetLayers) << "    PASSED" ;
    return true;
  } else {
//     edm::LogInfo(TkDetLayers) << "    FAILED " ;
    return false;
  }
}

float TIBLayer::computeWindowSize( const GeomDet* det, 
				   const TrajectoryStateOnSurface& tsos, 
				   const MeasurementEstimator& est) const
{
  // we assume the outer and inner rings have about same thickness...

//   edm::LogInfo(TkDetLayers) << "TIBLayer::computeWindowSize: Y axis of tangent plane is"
//        << plane.toGlobal( LocalVector(0,1,0)) ;

  MeasurementEstimator::Local2DVector localError( est.maximalLocalDisplacement(tsos, det->surface()));
  float yError = localError.y();

  float tanTheta = tan( tsos.globalMomentum().theta());
  float thickCorrection = det->surface().bounds().thickness() / (2.*fabs( tanTheta));

  // FIXME: correct this in case of wide phi window !  

  return yError + thickCorrection;
}



