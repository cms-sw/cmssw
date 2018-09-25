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

typedef GeometricSearchDet::DetWithState DetWithState;

TIBLayer::TIBLayer(std::vector<const TIBRing*>& innerRings,
                   std::vector<const TIBRing*>& outerRings)
  : TBLayer(innerRings,outerRings, GeomDetEnumerators::TIB)
{
  theComps.assign(theInnerComps.begin(),theInnerComps.end());
  theComps.insert(theComps.end(),theOuterComps.begin(),theOuterComps.end());
  
  std::sort(theComps.begin(),theComps.end(),isDetLessZ);
  std::sort(theInnerComps.begin(),theInnerComps.end(),isDetLessZ);
  std::sort(theOuterComps.begin(),theOuterComps.end(),isDetLessZ);
  
  for(auto & it : theComps)
  {
    theBasicComps.insert(theBasicComps.end(),
                         it->basicComponents().begin(),
                         it->basicComponents().end());
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

  for (auto & i : theInnerComps)
  {
    LogDebug("TkDetLayers") << "inner TIBRing pos z,radius,eta,phi: " 
                << i->position().z() << " , " 
                << i->position().perp() << " , " 
                << i->position().eta() << " , " 
                << i->position().phi() ;
  }

  for (auto & i : theOuterComps)
  {
    LogDebug("TkDetLayers") << "outer TIBRing pos z,radius,eta,phi: " 
                << i->position().z() << " , " 
                << i->position().perp() << " , " 
                << i->position().eta() << " , " 
                << i->position().phi() ;
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

TIBLayer::~TIBLayer(){} 

  


BoundCylinder* 
TIBLayer::cylinder( const std::vector<const GeometricSearchDet*>& rings)
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


std::tuple<bool,int,int>  TIBLayer::computeIndexes(GlobalPoint gInnerPoint, GlobalPoint gOuterPoint) const {

  int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.z());
  const GeometricSearchDet* innerRing =  theInnerComps[innerIndex];
  float innerDist = std::abs( innerRing->surface().position().z() - gInnerPoint.z());

  int outerIndex = theOuterBinFinder.binIndex(gOuterPoint.z());
  const GeometricSearchDet* outerRing = theOuterComps[outerIndex];
  float outerDist = std::abs( outerRing->surface().position().z() - gOuterPoint.z());


  return std::make_tuple(innerDist < outerDist,innerIndex, outerIndex);


}

void TIBLayer::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				const Propagator& prop,
				const MeasurementEstimator& est,
				const SubLayerCrossing& crossing,
				float window, 
                std::vector<DetGroup>& result,
				bool checkClosest) const
{
  const GlobalPoint& gCrossingPos = crossing.position();

  const std::vector<const GeometricSearchDet*>& sLayer( subLayer( crossing.subLayerIndex()));
 
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
			float window)
{
  float halfLength = 0.5f*det.surface().bounds().length();

//   edm::LogInfo(TkDetLayers) << " TIBLayer: checking ring with z " << det.position().z();

  return std::abs( crossPoint.z()-det.position().z()) < (halfLength + window);
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

  // float tanTheta = std::tan( tsos.globalMomentum().theta());
  auto gm = tsos.globalMomentum();
  auto cotanTheta = gm.z()/gm.perp();
  float thickCorrection = 0.5f*det->surface().bounds().thickness()*std::abs( cotanTheta);

  // FIXME: correct this in case of wide phi window !  

  return yError + thickCorrection;
}



