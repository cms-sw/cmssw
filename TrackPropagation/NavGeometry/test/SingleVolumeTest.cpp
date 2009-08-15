#include "TrackPropagation/NavGeometry/interface/NavVolume6Faces.h"
#include "TrackPropagation/NavGeometry/interface/NavPlane.h"

#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "TrackPropagation/NavGeometry/test/stubs/RandomPlaneGeneratorByAxis.h"
#include "TrackPropagation/NavGeometry/test/stubs/UniformMomentumGenerator.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/VolumeGeometry/interface/MagneticFieldProvider.h"
///
#include  "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"
///
#include <vector>

using namespace std;

class MyMagneticField : public MagneticField
{
 public:
  virtual GlobalVector inTesla ( const GlobalPoint& ) const {return GlobalVector(0,0,4);}
};

class ConstantMagneticFieldProvider4T : public MagneticFieldProvider<float> {
public:
  virtual LocalVectorType valueInTesla( const LocalPointType& p) const {return LocalVectorType(0,0,4.);}
};

class ConstantMagVolume4T : public MagVolume {
public:
  ConstantMagVolume4T( const PositionType& pos, const RotationType& rot, 
		       DDSolidShape shape, const MagneticFieldProvider<float> * mfp) :
    MagVolume( pos, rot, shape, mfp) {}
 
  virtual bool inside( const GlobalPoint& gp, double tolerance=0.) const {return true;}

  /// Access to volume faces
  virtual const std::vector<VolumeSide>& faces() const {
    static std::vector<VolumeSide> dummy;
    return dummy;}
};


NavPlane* navPlane( RandomPlaneGenerator::PlanePtr p) {
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
    float zSize = 10.;
    float xPos = 0.;
    float yPos = 0.;
    float zPos = 0.;

    GlobalPoint zMinus( xPos, yPos, zPos - zSize);
    GlobalPoint zPlus( xPos, yPos, zPos + zSize);
    GlobalPoint phiMinus( xPos - xSize, yPos, zPos);
    GlobalPoint phiPlus( xPos + xSize, yPos, zPos);
    GlobalPoint rMinus( xPos, yPos - ySize, zPos);
    GlobalPoint rPlus( xPos, yPos + ySize, zPos);

    RandomPlaneGenerator::PlanePtr zMinusPlane = planeGenerator(zMinus,globalZ);
    RandomPlaneGenerator::PlanePtr zPlusPlane  = planeGenerator(zPlus,globalZ);
    RandomPlaneGenerator::PlanePtr phiMinusPlane = planeGenerator(phiMinus,globalX);
    RandomPlaneGenerator::PlanePtr phiPlusPlane  = planeGenerator(phiPlus,globalX);
    RandomPlaneGenerator::PlanePtr rMinusPlane = planeGenerator(rMinus,globalY);
    RandomPlaneGenerator::PlanePtr rPlusPlane  = planeGenerator(rPlus,globalY);

    //ReferenceCountingPointer<BoundPlane> zMinusPlane = planeGenerator(zMinus,globalZ);
    //ReferenceCountingPointer<BoundPlane> zPlusPlane  = planeGenerator(zPlus,globalZ);
    //ReferenceCountingPointer<BoundPlane> phiMinusPlane = planeGenerator(phiMinus,globalX);
    //ReferenceCountingPointer<BoundPlane> phiPlusPlane  = planeGenerator(phiPlus,globalX);
    //ReferenceCountingPointer<BoundPlane> rMinusPlane = planeGenerator(rMinus,globalY);
    //ReferenceCountingPointer<BoundPlane> rPlusPlane  = planeGenerator(rPlus,globalY);

    //cout << " testing NavPlane constructor " << endl;
    //NavPlane MyTestNavPlane(rPlusPlane);
    //cout << " Succesful !!!! " << endl;

    GlobalPoint volumePos( xPos, yPos, zPos);
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

    for (vector<NavVolumeSide>::const_iterator iv=MyNavVolumeSides.begin(); iv!=MyNavVolumeSides.end(); iv++) {

      const Plane& plane = dynamic_cast<const Plane&>(iv->surface().surface());

      cout << "TEST: surface " 
	   << iv->surface().surface().position()
	   << " normal vector " << plane.normalVector() 
	   << " side " << iv->surfaceSide() 
	   << " face " << iv->globalFace() << endl;
    }
    
    try {

        ConstantMagneticFieldProvider4T theProvider;
	ConstantMagVolume4T theMagVolume( MagVolume::PositionType(0,0,0), MagVolume::RotationType(),
				    ddshapeless, &theProvider);
	NavVolume6Faces vol( volumePos, volumeRot, ddshapeless, MyNavVolumeSides, 0);
	
	UniformMomentumGenerator momentumGenerator;
	//MM: Added MyTestField needed for Analytical Propagator
	// and added MyTestField to AnalyticalPropagator and GlobalTrajectoryParameters initialisers
	MyMagneticField  MyTestField;
	AlgebraicSymMatrix C(5,1);
	C *= 0.01;
	CurvilinearTrajectoryError err(C);

	// AnalyticalPropagator propagator ( &MyTestField, alongMomentum );
	RKPropagatorInS propagator ( theMagVolume, alongMomentum ); 


	for (int i=0; i<10; i++) {
	    GlobalVector gStartMomentum( momentumGenerator());
	    cout << "Start momentum is " << gStartMomentum << endl;
	    GlobalTrajectoryParameters gtp( GlobalPoint(xPos, yPos, zPos),
					    gStartMomentum, -1, &MyTestField );
 
	    RandomPlaneGenerator::PlanePtr startingPlane =  planeGenerator(gtp.position(),
									   gtp.momentum());
	    const BoundPlane& sp(*startingPlane);
	    FreeTrajectoryState fts(gtp);
	    ///	    TSOS startingState( fts, err, sp);
	    TSOS startingState( gtp, err, sp);

	    NavVolume::Container nsc = vol.nextSurface( vol.toLocal( gtp.position()), 
							vol.toLocal( gtp.momentum()), -1);
	    cout << "nextSurface size " << nsc.size() << endl;

	    int itry = 0;
	    for (NavVolume::Container::const_iterator isur = nsc.begin(); isur!=nsc.end(); isur++) {
		TSOS state = isur->surface().propagate( propagator, startingState);
		if (!state.isValid()) {
		    ++itry;
		    continue;
		}
		if (isur->bounds().inside(state.localPosition())) {
		    cout << "Surface containing destination point found at try " << itry << endl;
		    cout << "TSOS at final destination : " << state << endl;
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

