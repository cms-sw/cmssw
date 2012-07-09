#include "TrackPropagation/NavGeometry/interface/NavVolume6Faces.h"
#include "TrackPropagation/NavGeometry/interface/NavPlane.h"

#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "TrackPropagation/NavGeometry/test/stubs/RandomPlaneGeneratorByAxis.h"
#include "TrackPropagation/NavGeometry/test/stubs/UniformMomentumGenerator.h"
#include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"


#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/VolumeGeometry/interface/MagneticFieldProvider.h"
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
  virtual const std::vector<VolumeSide>& faces() const {return theFaces;}
private:
  std::vector<VolumeSide> theFaces;
};





NavPlane* navPlane( ReferenceCountingPointer<BoundPlane> p) {
  return new NavPlane(p.get());
}

SurfaceOrientation::Side oppositeSide( SurfaceOrientation::Side side = SurfaceOrientation::onSurface) {
  if ( side == SurfaceOrientation::onSurface ) {
    return side; 
  } else {
    SurfaceOrientation::Side oppositeSide = ( side ==SurfaceOrientation::positiveSide ? SurfaceOrientation::negativeSide : SurfaceOrientation::positiveSide);
    return oppositeSide;
  } 
}


int main() 
{

  
  TFile* theFile;
  TH1F* HistDx;
  TH1F* HistDy;
  TH1F* HistDz;
  TH1F* HistDp;
  TH2F* HistXY;
  TH2F* HistXY2;

  theFile = new TFile("DoubleVolumeTest.root","RECREATE");
  theFile->SetCompressionLevel(2);
  HistDx = new TH1F("Dx","",500,-.01,.01);
  HistDy = new TH1F("Dy","",500,-.01,.01);
  HistDz = new TH1F("Dz","",500,-.01,.01);
  HistDp = new TH1F("Dp","",500,-.01,.01);
  HistXY = new TH2F("XY","",500,-15.,15.,500,-15.,35.);
  HistXY2 = new TH2F("XY2","",500,-15.,15.,500,-15.,35.);
  //  HistNsuccesRK = new TH1F("DE","",,-1.,1.);
  //  HistNsuccesOld = new TH1F("DE","",50,-1.,1.);

    typedef TrajectoryStateOnSurface  TSOS;

    RandomPlaneGeneratorByAxis planeGenerator;
    planeGenerator.setTilt(0.1);

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

    GlobalPoint volumePos( xPos, yPos, zPos);
    GlobalPoint volumePos2( xPos, yPos + 2 * ySize , zPos);
    Surface::RotationType volumeRot; // unit matrix

    vector<NavVolumeSide> MyNavVolumeSides;
    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( rMinusPlane), SurfaceOrientation::inner,
				       rMinusPlane->side(volumePos,0)));

    NavPlane* CommonSideP = navPlane( rPlusPlane);

    MyNavVolumeSides.push_back( NavVolumeSide(CommonSideP,  SurfaceOrientation::outer,
    				rPlusPlane->side(volumePos,0))); 
    //    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( rPlusPlane), SurfaceOrientation::outer,
    //				       rPlusPlane->side(volumePos,0)));

    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( zMinusPlane), SurfaceOrientation::zminus,
				       zMinusPlane->side(volumePos,0)));
    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( zPlusPlane), SurfaceOrientation::zplus,
				       zPlusPlane->side(volumePos,0)));

    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( phiMinusPlane), SurfaceOrientation::phiminus,
				       phiMinusPlane->side(volumePos,0)));
    MyNavVolumeSides.push_back( NavVolumeSide( navPlane( phiPlusPlane), SurfaceOrientation::phiplus,
				       phiPlusPlane->side(volumePos,0)));


    vector<NavVolumeSide> MyNavVolumeSides2;
    MyNavVolumeSides2.push_back( NavVolumeSide( CommonSideP , SurfaceOrientation::inner,
				       rPlusPlane->side(volumePos2,0)));
    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( rPlus2Plane), SurfaceOrientation::outer,
				       rPlus2Plane->side(volumePos2,0)));

    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( zMinusPlane), SurfaceOrientation::zminus,
				       zMinusPlane->side(volumePos2,0)));
    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( zPlusPlane), SurfaceOrientation::zplus,
				       zPlusPlane->side(volumePos2,0)));

    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( phiMinusPlane), SurfaceOrientation::phiminus,
				       phiMinusPlane->side(volumePos2,0)));
    MyNavVolumeSides2.push_back( NavVolumeSide( navPlane( phiPlusPlane), SurfaceOrientation::phiplus,
				       phiPlusPlane->side(volumePos2,0)));

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
        ConstantMagneticFieldProvider4T theProvider;
	ConstantMagVolume4T theMagVolume( MagVolume::PositionType(0,0,0), MagVolume::RotationType(),
				    ddshapeless, &theProvider);


	NavVolume6Faces vol( volumePos, volumeRot, ddshapeless, MyNavVolumeSides, 0);
	NavVolume6Faces vol2( volumePos2, volumeRot, ddshapeless, MyNavVolumeSides2, 0);
	
	cout << "check if starting point is inside volume 1 : " << vol.inside(GlobalPoint(xPos, yPos, zPos),0.1) << endl;
	cout << "check if starting point is inside volume 2 : " << vol2.inside(GlobalPoint(xPos, yPos, zPos),0.1) << endl;

	UniformMomentumGenerator momentumGenerator;
	//MM: Added MyTestField needed for Analytical Propagator
	// and added MyTestField to AnalyticalPropagator and GlobalTrajectoryParameters initialisers
	MyMagneticField  MyTestField;
	AlgebraicSymMatrix C(5,1);
	C *= 0.01;
	CurvilinearTrajectoryError err(C);

	AnalyticalPropagator propagator ( &MyTestField, alongMomentum );

	// Propagator* oldPropagator = new RKPropagatorInS( theMagVolume, alongMomentum);
       	RKPropagatorInS RKprop ( theMagVolume, alongMomentum );



	for (int i=0; i<30; i++) {
	    GlobalVector gStartMomentum( momentumGenerator());

	    cout << "************* NEXT TEST 'EVENT' *****" << endl;
	    //	    	    cout << "Start momentum is " << gStartMomentum << endl;;

	    GlobalTrajectoryParameters gtp( GlobalPoint(xPos, yPos, zPos),
					    gStartMomentum, -1, &MyTestField );
 
	    RandomPlaneGenerator::PlanePtr startingPlane =  planeGenerator(gtp.position(),
									   gtp.momentum());
	    const BoundPlane& sp(*startingPlane);
	    FreeTrajectoryState fts(gtp);
	    TSOS startingState( gtp, err, sp);

	    TSOS CurrentRKState = startingState;
	    TSOS FinalAnalyticalState = startingState;
	    
	    //
	    ////
	    ////// TEST CrossToNextVolume Function:
	    ////
	    //

	    //
	    ///	    VolumeCrossReturnType is std::pair<const NavVolume*, TrajectoryStateOnSurface> 
	    //



	    VolumeCrossReturnType VolumeCrossResult = vol.crossToNextVolume(startingState, RKprop);

	    
	    CurrentRKState = VolumeCrossResult.tsos();
	    cout << "Current RK State is " << CurrentRKState << endl;
	    
	    double TotalPathLength = VolumeCrossResult.path();
	    //cout << "Oh pathlenght was : " << TotalPathLength << endl;
	    //	    if (TotalPathLength > 20) 
	    //  HistXY->Fill(CurrentRKState.globalPosition().x(),CurrentRKState.globalPosition().y());

	    if (TotalPathLength > 20) 
	      HistXY2->Fill(CurrentRKState.globalPosition().x(),CurrentRKState.globalPosition().y());
	    else
	      HistXY->Fill(CurrentRKState.globalPosition().x(),CurrentRKState.globalPosition().y());
	    
	    if (VolumeCrossResult.volume() != 0) {
	      //cout << "YES !!! Found next volume with position: " << VolumeCrossResult.volume()->position() << endl;
	      cout << "Do a second Iteration step !" << endl;

	      VolumeCrossReturnType VolumeCrossResult2 = VolumeCrossResult.volume()->crossToNextVolume(VolumeCrossResult.tsos(), RKprop);
	      if (VolumeCrossResult2.volume() != 0) {
		//		cout << "crossToNextVolume: Succeeded to find THIRD volume with pos, mom, " << 
		//  VolumeCrossResult2.tsos().globalPosition() << ", " << VolumeCrossResult2.tsos().globalMomentum() << endl;
		cout << "crossToNextVolume: Succeeded to find THIRD volume with state " << 
		  VolumeCrossResult2.tsos().globalPosition() << endl;
	      } else {
		//cout << "crossToNextVolume: Failed to find THIRD volume " << endl;
	      }
	      CurrentRKState = VolumeCrossResult2.tsos();	      
	      TotalPathLength += VolumeCrossResult2.path();
	      if (TotalPathLength > 20) 
		HistXY2->Fill(CurrentRKState.globalPosition().x(),CurrentRKState.globalPosition().y());
	      else
		HistXY->Fill(CurrentRKState.globalPosition().x(),CurrentRKState.globalPosition().y());
	    } else {
	      //cout << "crossToNextVolume: NO !!!!!!!! Nothing on other side" << endl;
	    }
	    
	    cout << " - - - - - - That was crossToNextVolume, now compare to conventional result: " << endl;



	    //
	    ////
	    ////// OK 


	    NavVolume::Container nsc = vol.nextSurface( vol.toLocal( gtp.position()), 
							vol.toLocal( gtp.momentum()), -1);
	    // cout << "nextSurface size " << nsc.size() << endl;


	    int itry = 0;
	    const NavVolume* nextVol = 0;
	    const Surface* exitSurface = 0;

	    for (NavVolume::Container::const_iterator isur = nsc.begin(); isur!=nsc.end(); isur++) {
		TSOS state = isur->surface().propagate( propagator, startingState);
		if (!state.isValid()) {
		    ++itry;
		    continue;
		}
		if (isur->bounds().inside(state.localPosition())) {
		  //		    cout << "Surface containing destination point found at try " << itry << endl;
		    nextVol = isur->surface().nextVolume(state.localPosition(),oppositeSide(isur->side()));

		    //cout << "Looking for next Volume on other side of surface with center " << isur->surface().surface().position() << endl;
		    startingState = state;
		    FinalAnalyticalState = state;
		    exitSurface = &( isur->surface().surface() );
		    break;
		}
		else {
		    ++itry;
		}
	    }
	    

	    cout << "And here analytical State is " << FinalAnalyticalState << endl;

	    if (nextVol != 0) {
	      //cout << "YES !!! Found next volume with position: " << nextVol->position() << endl;
	      cout << "Do a second Iteration step !" << endl;
		
	      //	      NavVolume::Container nsc2 = nextVol->nextSurface( nextVol->toLocal( startingState.globalPosition()+0.01*startingState.globalMomentum()), 
	      //					nextVol->toLocal( startingState.globalMomentum()), -1);
	      NavVolume::Container nsc2 = nextVol->nextSurface( nextVol->toLocal( startingState.globalPosition()), 
							nextVol->toLocal( startingState.globalMomentum()), -1, alongMomentum, exitSurface );

	      
	      for (NavVolume::Container::const_iterator isur = nsc2.begin(); isur!=nsc2.end(); isur++) {
		TSOS state = isur->surface().propagate( propagator, startingState);
		if (!state.isValid()) {
		  ++itry;
		  continue;
		}
		if (isur->bounds().inside(state.localPosition())) {
		  //  cout << "** SECOND ** Surface containing destination point found at try " << itry << endl;
		  //cout << "Position and momentum after first step: " << startingState.globalPosition() << ", " << startingState.globalMomentum() << endl;	
		  //cout << "Position and momentum after second step: " << state.globalPosition() << ", " << state.globalMomentum() << endl;

		  nextVol = isur->surface().nextVolume(state.localPosition(),oppositeSide(isur->side()));
		  
		  // cout << "Looking for ** NEXT ** next Volume on other side of surface with center " << isur->surface().surface().position() << endl;
		  FinalAnalyticalState = state;
		  break;
		}
		else {
		  ++itry;
		}
	      }
	      if (nextVol != 0) {
		//		cout << "Succeeded to find THIRD volume with pos, mom, " << startingState.globalPosition() << ", " << startingState.globalMomentum() << endl;
		cout << "Succeeded to find THIRD volume with state " << startingState << endl;
	      } else {
		//cout << "Failed to find THIRD volume " << endl;
	      }
	    } else {
	      //cout << "NO !!!!!!!! Nothing on other side" << endl;

	      
	    }
	    
	    HistDx->Fill(1000.*(CurrentRKState.globalPosition().x()-FinalAnalyticalState.globalPosition().x()));
	    HistDy->Fill(1000.*(CurrentRKState.globalPosition().y()-FinalAnalyticalState.globalPosition().y()));
	    HistDz->Fill(1000.*(CurrentRKState.globalPosition().z()-FinalAnalyticalState.globalPosition().z()));
	    HistDp->Fill(1000.*(CurrentRKState.globalMomentum().mag()-FinalAnalyticalState.globalMomentum().mag()));


	}
    }
    catch (std::exception& ex) {
	cout << "Oops, got an exception: " << ex.what() << endl;
	return 1;
    }
    HistDx->Write();
    HistDy->Write();
    HistDz->Write();
    HistDp->Write();
    HistXY->Write();
    HistXY2->Write();
    theFile->Close();
    //    HistNsuccesOld->Write();
    //    HistNsuccesRK->Write();
}

