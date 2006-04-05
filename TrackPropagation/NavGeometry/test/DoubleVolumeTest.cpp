#include "TrackPropagation/NavGeometry/interface/NavVolume6Faces.h"
#include "TrackPropagation/NavGeometry/interface/NavPlane.h"

#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

#include "TrackPropagation/NavGeometry/test/stubs/RandomPlaneGeneratorByAxis.h"
#include "TrackPropagation/NavGeometry/test/stubs/UniformMomentumGenerator.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include <vector>

using namespace std;

class MyMagneticField : public MagneticField
{
 public:
  virtual GlobalVector inTesla ( const GlobalPoint& ) const {return GlobalVector(0,0,4);}
};





NavPlane* navPlane( ReferenceCountingPointer<BoundPlane> p) {
  return new NavPlane(p.get());
}



int main() 
{
    typedef TrajectoryStateOnSurface  TSOS;

    RandomPlaneGeneratorByAxis planeGenerator;
    planeGenerator.setTilt(0.00);

    GlobalVector globalX(1,0,0);
    GlobalVector globalY(0,1,0);
    GlobalVector globalZ(0,0,1);


    float xSize = 10.;
    float ySize = 10.;
    float zSize = 100.;
    float xPos = 0.;
    float yPos = 0.;
    float zPos = 0.;

    GlobalPoint zMinus( xPos, yPos, zPos - zSize);
    GlobalPoint zPlus( xPos, yPos, zPos + zSize);
    GlobalPoint phiMinus( xPos - xSize, yPos, zPos);
    GlobalPoint phiPlus( xPos + xSize, yPos, zPos);
    GlobalPoint rMinus( xPos, yPos - ySize, zPos);
    GlobalPoint rPlus( xPos, yPos + ySize, zPos);
    GlobalPoint rPlus2( xPos, yPos + 3 * ySize, zPos);

    //RandomPlaneGenerator::PlanePtr zMinusPlane = planeGenerator(zMinus,globalZ);
    //RandomPlaneGenerator::PlanePtr zPlusPlane  = planeGenerator(zPlus,globalZ);
    //RandomPlaneGenerator::PlanePtr phiMinusPlane = planeGenerator(phiMinus,globalX);
    //RandomPlaneGenerator::PlanePtr phiPlusPlane  = planeGenerator(phiPlus,globalX);
    //RandomPlaneGenerator::PlanePtr rMinusPlane = planeGenerator(rMinus,globalY);
    //RandomPlaneGenerator::PlanePtr rPlusPlane  = planeGenerator(rPlus,globalY);

    ReferenceCountingPointer<BoundPlane> zMinusPlane = planeGenerator(zMinus,globalZ);
    ReferenceCountingPointer<BoundPlane> zPlusPlane  = planeGenerator(zPlus,globalZ);
    ReferenceCountingPointer<BoundPlane> phiMinusPlane = planeGenerator(phiMinus,globalX);
    ReferenceCountingPointer<BoundPlane> phiPlusPlane  = planeGenerator(phiPlus,globalX);
    ReferenceCountingPointer<BoundPlane> rMinusPlane = planeGenerator(rMinus,globalY);
    ReferenceCountingPointer<BoundPlane> rPlusPlane  = planeGenerator(rPlus,globalY);
    ReferenceCountingPointer<BoundPlane> rPlus2Plane  = planeGenerator(rPlus2,globalY);

    //cout << " testing NavPlane constructor " << endl;
    //NavPlane MyTestNavPlane(rPlusPlane);
    //cout << " Succesful !!!! " << endl;

    GlobalPoint volumePos( xPos, yPos, zPos);
    GlobalPoint volumePos2( xPos, yPos + 2 * ySize , zPos);
    Surface::RotationType volumeRot; // unit matrix

    vector<NavVolumeSide> MyNavVolumeSides;
    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( rMinusPlane), SurfaceOrientation::inner,
				       rMinusPlane->side(volumePos,0)));
    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( rPlusPlane), SurfaceOrientation::outer,
				       rPlusPlane->side(volumePos,0)));

    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( zMinusPlane), SurfaceOrientation::zminus,
				       zMinusPlane->side(volumePos,0)));
    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( zPlusPlane), SurfaceOrientation::zplus,
				       zPlusPlane->side(volumePos,0)));

    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( phiMinusPlane), SurfaceOrientation::phiminus,
				       phiMinusPlane->side(volumePos,0)));
    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( phiPlusPlane), SurfaceOrientation::phiplus,
				       phiPlusPlane->side(volumePos,0)));
    cout << "MIDDLE of the NavVolume: " << volumePos << endl;
    cout << "... but rotated using rotation: " << endl; cout << volumeRot << endl;


    vector<NavVolumeSide> MyNavVolumeSides2;
    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( rPlusPlane), SurfaceOrientation::inner,
				       rMinusPlane->side(volumePos,0)));
    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( rPlus2Plane), SurfaceOrientation::outer,
				       rPlusPlane->side(volumePos,0)));

    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( zMinusPlane), SurfaceOrientation::zminus,
				       zMinusPlane->side(volumePos,0)));
    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( zPlusPlane), SurfaceOrientation::zplus,
				       zPlusPlane->side(volumePos,0)));

    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( phiMinusPlane), SurfaceOrientation::phiminus,
				       phiMinusPlane->side(volumePos,0)));
    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( phiPlusPlane), SurfaceOrientation::phiplus,
				       phiPlusPlane->side(volumePos,0)));
    cout << "MIDDLE of the NavVolume2: " << volumePos2 << endl;
    cout << "... but rotated using rotation: " << endl; cout << volumeRot << endl;





    for (vector<NavVolumeSide>::const_iterator iv=MyNavVolumeSides.begin(); iv!=MyNavVolumeSides.end(); iv++) {

      const Plane& plane = dynamic_cast<const Plane&>(iv->surface().surface());

      cout << "TEST: surface " 
	//   << iv->volumeSides.begin() 
        //   << " at position "
	   << iv->surface().surface().position()
	//   << " and with rotation "
	//   << iv->surface().surface().rotation() 

	   << " normal vector " << plane.normalVector() 
	   << " side " << iv->surfaceSide() 
	   << " face " << iv->globalFace() << endl;

    }
    
    try {
	NavVolume6Faces vol( volumePos, volumeRot, ddshapeless, MyNavVolumeSides, 0);
	


	NavVolume6Faces vol2( volumePos, volumeRot, ddshapeless, MyNavVolumeSides2, 0);
	
	cout << "check if starting point is inside volume 1 : " << vol.inside(GlobalPoint(xPos, yPos, zPos),0.1) << endl;
	cout << "check if starting point is inside volume 2 : " << vol2.inside(GlobalPoint(xPos, yPos, zPos),0.1) << endl;

	UniformMomentumGenerator momentumGenerator;
	//MM: Added MyTestField needed for Analytical Propagator
	// and added MyTestField to AnalyticalPropagator and GlobalTrajectoryParameters initialisers
	MyMagneticField  MyTestField;
	AnalyticalPropagator propagator ( &MyTestField, alongMomentum );

	for (int i=0; i<10; i++) {
	    GlobalVector gStartMomentum( momentumGenerator());

	    cout << "************* " << endl;
	    //	    cout << "Start momentum is " << gStartMomentum << endl;;

	    GlobalTrajectoryParameters gtp( GlobalPoint(xPos, yPos, zPos),
					    gStartMomentum, -1, &MyTestField );
 
	    RandomPlaneGenerator::PlanePtr startingPlane =  planeGenerator(gtp.position(),
									   gtp.momentum());
	    const BoundPlane& sp(*startingPlane);
	    FreeTrajectoryState fts(gtp);
	    TSOS startingState( fts, sp);

	    NavVolume::Container nsc = vol.nextSurface( vol.toLocal( gtp.position()), 
							vol.toLocal( gtp.momentum()), -1);
	    // cout << "nextSurface size " << nsc.size() << endl;

	    

	    int itry = 0;

	    for (NavVolume::Container::const_iterator isur = nsc.begin(); isur!=nsc.end(); isur++) {
		TSOS state = isur->first->propagate( propagator, startingState);
		if (!state.isValid()) {
		    ++itry;
		    continue;
		}
		if (isur->second->inside(state.localPosition())) {
		    cout << "Surface containing destination point found at try " << itry << endl;

		    // Write out some information: 
		    cout << "and position is now " << state.globalPosition() << endl;
		    //
		    /* isur->first is a pointer to a NavSurface, ->second points at a Bounds */
		    //
		    cout << "which is on the surface with center " << isur->first->surface().position() << endl;
		    //		
		    cout << "and gives acces to volume " << endl;
		    cout << " [ ... still trying to find out how to get to nextVolume ... ] " << endl;
		    // isur->first->nextVolume(state.localPosition(),isur->first->surface().position()) << endl;
		    int ivol = 0;
		    for (vector<NavVolumeSide>::const_iterator iv=MyNavVolumeSides.begin(); iv!=MyNavVolumeSides.end(); iv++) {
		      ivol++;
		      cout << " ... trying side no. " << ivol << endl;
		      //     if (iv->surface().surface().position() == isur->first->surface().position()) {
		      //cout << "Matched to Volumside no. " << ivol << endl;
		      //}
		    }
		    break;
		}
		else {
		    ++itry;
		}
	    }	       
	}

    }
    catch (std::exception& ex) {
	cout << "Oops, got an exception: " << ex.what() << endl;
	return 1;
    }

    
}

