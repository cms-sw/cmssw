#include "TrackPropagation/NavGeometry/interface/NavVolume6Faces.h"
#include "MagneticField/VolumeGeometry/interface/FourPointPlaneBounds.h"
#include "TrackPropagation/NavGeometry/src/ThreePlaneCrossing.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "DataFormats/GeometrySurface/interface/GeneralNSurfaceDelimitedBounds.h"
#include "TrackPropagation/NavGeometry/interface/NavSurface.h"
#include "TrackPropagation/NavGeometry/interface/NavSurfaceBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <map>

SurfaceOrientation::Side oppositeSide( SurfaceOrientation::Side side = SurfaceOrientation::onSurface) {
  if ( side == SurfaceOrientation::onSurface ) {
    return side; 
  } else {
    SurfaceOrientation::Side oppositeSide = ( side ==SurfaceOrientation::positiveSide ? SurfaceOrientation::negativeSide : SurfaceOrientation::positiveSide);
    return oppositeSide;
  } 
}


NavVolume6Faces::NavVolume6Faces( const PositionType& pos,
				  const RotationType& rot, 
				  DDSolidShape shape,
				  const std::vector<NavVolumeSide>& faces,
				  const MagneticFieldProvider<float> * mfp) :
  NavVolume(pos,rot,shape,mfp) 
{
  for (std::vector<NavVolumeSide>::const_iterator i=faces.begin();
       i != faces.end(); i++) {
    theFaces.push_back( VolumeSide( const_cast<Surface*>(&(i->surface().surface())), 
				    i->globalFace(), i->surfaceSide()));
    //  LogDebug("NavGeometry") << " or actually this is where we have side " << i->surfaceSide() << " and face " << i->globalFace() ;
  }



    computeBounds(faces);
}

NavVolume6Faces::NavVolume6Faces( const MagVolume& magvol, bool isIron) :
  NavVolume( magvol.position(), magvol.rotation(), magvol.shapeType(), magvol.provider()),
  theFaces(magvol.faces()), isThisIron(isIron)
{
  std::vector<NavVolumeSide> navSides;
  std::vector<VolumeSide> magSides( magvol.faces());
  NavSurfaceBuilder navBuilder;

  for (std::vector<VolumeSide>::const_iterator i=magSides.begin();
       i != magSides.end(); i++) {
    NavSurface* navSurface = navBuilder.build( i->surface());
    navSides.push_back( NavVolumeSide( navSurface, i->globalFace(), i->surfaceSide()));
  }
  computeBounds(navSides);
}


bool NavVolume6Faces::inside( const GlobalPoint& gp, double tolerance) const 
{
  // check if the point is on the correct side of all delimiting surfaces
  for (std::vector<VolumeSide>::const_iterator i=theFaces.begin(); i!=theFaces.end(); i++) {
    Surface::Side side = i->surface().side( gp, tolerance);
    if ( side != i->surfaceSide() && side != SurfaceOrientation::onSurface) return false;
  }
  return true;
}



void NavVolume6Faces::computeBounds(const std::vector<NavVolumeSide>& faces) 
{
    bool allPlanes = true;
    // bool allPlanes = false; // for TESTS ONLY!!!
    std::vector<const Plane*> planes;
    for (std::vector<NavVolumeSide>::const_iterator iface=faces.begin(); iface!=faces.end(); iface++) {
	const Plane* plane = dynamic_cast<const Plane*>(&(iface->surface()));
	if (plane != nullptr) {
	    planes.push_back(plane);
	}
	else allPlanes = false;
    }
    
    for (unsigned int i=0; i<faces.size(); i++) {

      // FIXME: who owns the new NavSurface? memory leak???

	NavSurface& navSurf = faces[i].mutableSurface();
	Bounds* myBounds = nullptr;
	if (allPlanes) {	
	    myBounds = computeBounds( i, planes);
	}
	else {
	    myBounds = computeBounds( i, faces);
	}
	navSurf.addVolume( this, myBounds, faces[i].surfaceSide());
	delete myBounds; // since navSurf now owns a copy

// this is tricky: we want to avoid multiple copies of the Bounds; the NavSurface owns
// a copy of Bounds for each touching Volume (instantiated in the call to addVolume).
// We would like to keep a pointer to the same Bounds in the NavVolume, so we have to ASK
// the NavSurface for the Bounds* of the Bounds we just gave it!
	//LogDebug("NavGeometry") << "Adding a Volume Side with center " << navSurf.surface().position() << " side "<< faces[i].surfaceSide() << " and face " << faces[i].globalFace();
	theNavSurfaces.push_back( SurfaceAndBounds(&navSurf, navSurf.bounds(this), faces[i].surfaceSide(), faces[i].globalFace()));
    }
}

Bounds* NavVolume6Faces::computeBounds( int index, 
					const std::vector<const Plane*>& bpc)
{
  const Plane* plane( bpc[index]);

  // find the 4 intersecting planes
  int startIndex = 2*(1+index/2); // 2, 4, 6
  std::vector<const Plane*> crossed; crossed.reserve(4);
  for (int j = startIndex; j <  startIndex+4; j++) {
    crossed.push_back(bpc[j%6]);
  }

  // compute intersection corners of the plane triplets
  std::vector<GlobalPoint> corners; corners.reserve(4);
  ThreePlaneCrossing crossing;
  for ( int i=0; i<2; i++) {
    for ( int j=2; j<4; j++) {
      GlobalPoint corner( crossing.crossing( *plane, *crossed[i], *crossed[j]));
      corners.push_back(corner);

#ifdef DEBUG
      std::cout << "Crossing of planes is " << corner << std::endl;
      std::cout << "NormalVectors of the planes are " << plane->normalVector()
	   << " " << crossed[i]->normalVector() << " " << crossed[j]->normalVector() << std::endl;
      std::cout << "Positions of planes are " << plane->position()
	   << " " << crossed[i]->position() << " " << crossed[j]->position() << std::endl;
      if (plane->side( corner, 1.e-5) == SurfaceOrientation::onSurface &&
	  crossed[i]->side( corner, 1.e-5) == SurfaceOrientation::onSurface &&
	  crossed[j]->side( corner, 1.e-5) == SurfaceOrientation::onSurface) {
	  std::cout << "Crossing is really on all three surfaces" << std::endl;
      }
      else {
	  std::cout << "CROSSING IS NOT ON SURFACES!!!" << std::endl;
	  std::cout << plane->localZ(corner) << std::endl;
	  std::cout << crossed[i]->localZ(corner) << std::endl;
	  std::cout << crossed[j]->localZ(corner) << std::endl;
       }
#endif

    }
  }

  // put corners in cyclic sequence (2 and 3 swapped)
  return new FourPointPlaneBounds( plane->toLocal( corners[0]), plane->toLocal( corners[1]),
 				   plane->toLocal( corners[3]), plane->toLocal( corners[2]));
}

Bounds* NavVolume6Faces::computeBounds( int index, const std::vector<NavVolumeSide>& faces)
{
    typedef GeneralNSurfaceDelimitedBounds::SurfaceAndSide         SurfaceAndSide;
    typedef GeneralNSurfaceDelimitedBounds::SurfaceContainer       SurfaceContainer;

  // find the 4 intersecting surfaces
  int startIndex = 2*(1+index/2); // 2, 4, 6
  SurfaceContainer crossed; crossed.reserve(4);
  for (int j = startIndex; j <  startIndex+4; j++) {
    const NavVolumeSide& face(faces[j%6]);
    crossed.push_back( SurfaceAndSide(&(face.surface().surface()), face.surfaceSide()));
  }
  return new GeneralNSurfaceDelimitedBounds( &(faces[index].surface().surface()), crossed);
}


NavVolume::Container
NavVolume6Faces::nextSurface( const NavVolume::LocalPoint& pos, 
			      const NavVolume::LocalVector& mom,
			      double charge, PropagationDirection propDir) const
{
    typedef std::map<double,SurfaceAndBounds>   SortedContainer;

    GlobalPoint  gpos( toGlobal(pos));
    GlobalVector gmom( toGlobal(mom));
    GlobalVector gdir = (propDir == alongMomentum ? gmom : -gmom);

    SortedContainer sortedSurfaces;
    Container       verycloseSurfaces; // reachable surface with dist < epsilon !!
    Container       unreachableSurfaces;

    double epsilon = 0.01; // should epsilon be hard-coded or a variable in NavVolume?

    for (Container::const_iterator i=theNavSurfaces.begin(); i!=theNavSurfaces.end(); i++) {
	std::pair<bool,double> dist = i->surface().distanceAlongLine( gpos, gdir);
	if (dist.first) {
	  if (dist.second > epsilon) sortedSurfaces[dist.second] = *i;
	  else verycloseSurfaces.push_back(*i);
	} 
	else unreachableSurfaces.push_back(*i);
    }
    NavVolume::Container result;
    for (SortedContainer::const_iterator i=sortedSurfaces.begin(); i!=sortedSurfaces.end(); ++i) {
	result.push_back(i->second);
    }
    result.insert( result.end(), unreachableSurfaces.begin(), unreachableSurfaces.end());
    result.insert( result.end(), verycloseSurfaces.begin(), verycloseSurfaces.end());
    return result;
}

NavVolume::Container
NavVolume6Faces::nextSurface( const NavVolume::LocalPoint& pos, 
			      const NavVolume::LocalVector& mom,
			      double charge, PropagationDirection propDir,
			      ConstReferenceCountingPointer<Surface> NotThisSurfaceP) const
{
    typedef std::map<double,SurfaceAndBounds>   SortedContainer;

    GlobalPoint  gpos( toGlobal(pos));
    GlobalVector gmom( toGlobal(mom));
    GlobalVector gdir = (propDir == alongMomentum ? gmom : -gmom);

    SortedContainer sortedSurfaces;
    Container       verycloseSurfaces; // reachable surface with dist < epsilon (if 6-surface check fails)
    Container       unreachableSurfaces;

    double epsilon = 0.01; // should epsilon be hard-coded or a variable in NavVolume?
    bool surfaceMatched = false;

    for (Container::const_iterator i=theNavSurfaces.begin(); i!=theNavSurfaces.end(); i++) {
      if (&(i->surface().surface()) == NotThisSurfaceP) surfaceMatched = true;
    }
    
    for (Container::const_iterator i=theNavSurfaces.begin(); i!=theNavSurfaces.end(); i++) {
	std::pair<bool,double> dist = i->surface().distanceAlongLine( gpos, gdir);
	if (dist.first) { 
	  if ( &(i->surface().surface()) == NotThisSurfaceP || (!surfaceMatched && dist.second < epsilon)) 
	    verycloseSurfaces.push_back(*i);
	  else sortedSurfaces[dist.second] = *i;
	}
	else unreachableSurfaces.push_back(*i);
    }

    NavVolume::Container result;
    for (SortedContainer::const_iterator i=sortedSurfaces.begin(); i!=sortedSurfaces.end(); ++i) {
	result.push_back(i->second);
    }
    result.insert( result.end(), unreachableSurfaces.begin(), unreachableSurfaces.end());
    result.insert( result.end(), verycloseSurfaces.begin(), verycloseSurfaces.end());
    return result;
}

VolumeCrossReturnType
NavVolume6Faces::crossToNextVolume( const TrajectoryStateOnSurface& startingState, const Propagator& prop ) const
{
  typedef TrajectoryStateOnSurface TSOS;
  typedef std::pair<TSOS,double> TSOSwithPath;

  NavVolume::Container nsc = nextSurface( toLocal( startingState.globalPosition()), 
					  toLocal( startingState.globalMomentum()), -1,
					  alongMomentum, &(startingState.surface()));
  int itry = 0;
  VolumeCrossReturnType VolumeCrossResult( nullptr, startingState, 0.0);

  for (NavVolume::Container::const_iterator isur = nsc.begin(); isur!=nsc.end(); isur++) {

    LogDebug("NavGeometry") <<  "crossToNextVolume: trying Surface no. " << itry ;
    TSOSwithPath state;
    
    try {
      state = isur->surface().propagateWithPath( prop, startingState);
    }
    catch (MagVolumeOutsideValidity& except) {
      LogDebug("NavGeometry") << "Ohoh... failed to stay inside magnetic field !! skip this surface " ;
	++itry;
      continue;
    }
    
    if (!state.first.isValid()) {
      ++itry;
      continue;
    }

    LogDebug("NavGeometry") <<  "crossToNextVolume: reached Valid State at Surface no. " << itry ;
    LogDebug("NavGeometry") << " --> local position of Valid state is " <<  state.first.localPosition()  ;
    LogDebug("NavGeometry") << " --> global position of Valid state is " <<  state.first.globalPosition()  ;

    if (isur->bounds().inside(state.first.localPosition())) {
      //LogDebug("NavGeometry") << "crossToNextVolume: Surface containing destination point found at try " << itry ;
      // Found the exit surface !! Get pointer to next volume and save exit state:
      //VolumeCrossResult.first = isur->surface().nextVolume(state.localPosition(),oppositeSide(isur->side()));
      //VolumeCrossResult.second = state;
      //      exitSurface = &( isur->surface().surface() );
      //if(VolumeCrossResult.path() < 0.01) {
      //	LogDebug("NavGeometry") << " Stuck at  " << state.first.globalPosition() ;
      //}
      return VolumeCrossReturnType ( isur->surface().nextVolume(state.first.localPosition(), oppositeSide(isur->side())),
			    state.first, state.second );
      
      break;
    }
    else {
      LogDebug("NavGeometry") << "crossToNextVolume: BUT not inside the Bounds !! " ;
      ++itry;
    }
  }

  return VolumeCrossResult;
}

/*
std::pair<bool,double>
NavVolume6Faces::linearDistance( const NavSurface& surf, const NavVolume::LocalPoint& pos, 
                                 const NavVolume::LocalVector& mom) const
{
    const Plane* plane = dynamic_cast<const Plane*>(&surf);
    if (plane != 0) {
	

}
*/

/*
NavVolume::Container
NavVolume6Faces::nextSurface( const NavVolume::LocalPoint& pos, 
			      const NavVolume::LocalVector& mom,
			      double charge, PropagationDirection propDir) const
{
    StraightLinePlaneCrossing pc( toGlobal(pos).basicVector(), toGlobal(mom).basicVector(), propDir);
    Container approaching;
    Container movingaway;
    SurfaceAndBounds bestGuess;

    for (Container::const_iterator i=theNavSurfaces.begin(); i!=theNavSurfaces.end(); i++) {
	const Plane& plane = dynamic_cast<const Plane&>(*(i->first));
	std::pair<bool,StraightLinePlaneCrossing::PositionType> crossed = pc.position( plane);
	if (crossed.first) {

#ifdef DEBUG
	    std::cout << "Plane crossed at global point " << crossed.second
		 << " local point " << plane.toLocal( Plane::GlobalPoint(crossed.second)) << std::endl;
#endif

	    if ( i->second->inside( plane.toLocal( Plane::GlobalPoint(crossed.second)))) {
		bestGuess = SurfaceAndBounds( i->first, i->second);
	    }
	    else {
		// momentm is pointing towards the plane
		approaching.push_back( SurfaceAndBounds( i->first, i->second));
	    }
	}
	else {
	    movingaway.push_back( SurfaceAndBounds( i->first, i->second));
	}
    }

    NavVolume::Container result(1,bestGuess); result.reserve(theNavSurfaces.size());
    result.insert(result.end(), approaching.begin(), approaching.end());
    result.insert(result.end(), movingaway.begin(), movingaway.end());
    return result;
}
*/
