#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>
#include <string>
#include <map>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

////////////////////////////////////////////////////////////////////////////

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

////////////////////////////////////////////////////////////////////////////

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/RungeKutta/interface/RKTestPropagator.h"


using namespace std;

class RKTest : public edm::EDAnalyzer {
public:
  RKTest(const edm::ParameterSet& pset) {}

  ~RKTest(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){
    using namespace edm;
    ESHandle<MagneticField> magfield;
    setup.get<IdealMagneticFieldRecord>().get(magfield);

    propagateInCentralVolume( &(*magfield));
  }

private:

  typedef TrajectoryStateOnSurface TSOS;

  void propagateInCentralVolume( const MagneticField* field) const;
  Surface::RotationType rotation( const GlobalVector& zAxis) const;

};



void RKTest::propagateInCentralVolume( const MagneticField* field) const
{

  // RKTestField is used internally in RKTestPropagator
  // In the following code "field" or "&TestField" are interchangeable
  // They should give identical field values if "field" is produced by VolumeBasedMagneticField
  // with the non-default option "useParametrizedTrackerField = true"
  
  RKTestField TestField; // Not needed if you want to use field instead of &TestField

  //  RKTestPropagator RKprop ( &TestField, alongMomentum );
  //AnalyticalPropagator ANprop  ( &TestField, alongMomentum);
  RKTestPropagator RKprop ( field, alongMomentum );
  AnalyticalPropagator ANprop  ( field, alongMomentum);


  for (float phi = -3.14; phi<3.14 ; phi+=0.5) {
    for (float costh = -0.99; costh<+0.99 ; costh+=0.3) {
      cout << "And now trying costh, phi = " << costh << ", " << phi << endl;
      
      //Define starting position and momentum
      float sinth = sqrt(1-costh*costh);
      GlobalVector startingMomentum(5*sin(phi)*sinth,5*cos(phi)*sinth,5*costh);
      GlobalPoint startingPosition(0,0,0);
      //Define starting plane
      PlaneBuilder pb;
      Surface::RotationType rot = rotation( startingMomentum);
      PlaneBuilder::ReturnType startingPlane = pb.plane( startingPosition, rot);
      // Define end plane
      float propDistance = 100; // 100 cm
      GlobalPoint targetPos( (propDistance*startingMomentum.unit()).basicVector());
      PlaneBuilder::ReturnType EndPlane = pb.plane( targetPos, rot);
      // Define error matrix
      AlgebraicSymMatrix C(5,1);
      C *= 0.01;
      CurvilinearTrajectoryError err(C);

      TSOS startingStateP( GlobalTrajectoryParameters(startingPosition, 
						      startingMomentum, 1, &TestField), 
			   err, *startingPlane);

      TSOS startingStateM( GlobalTrajectoryParameters(startingPosition, 
						      startingMomentum, -1, &TestField), 
			   err, *startingPlane);

      try {
	TSOS trackStateP = RKprop.propagate( startingStateP, *EndPlane);
	cout << "Succesfully finished Positive track propagation  -------------- with RK: " << trackStateP.globalPosition() << endl;
	TSOS trackStateP2 = ANprop.propagate( startingStateP, *EndPlane);
	cout << "Succesfully finished Positive track propagation  -------------- with AN: " << trackStateP2.globalPosition() << endl;
      } catch (MagVolumeOutsideValidity & duh){
	cout << "MagVolumeOutsideValidity not properly caught!! Lost this track " << endl;
      }

      try {
	TSOS trackStateM = RKprop.propagate( startingStateM, *EndPlane);
	cout << "Succesfully finished Negative track propagation  -------------- with RK: " << trackStateM.globalPosition() << endl;
	TSOS trackStateM2 = ANprop.propagate( startingStateM, *EndPlane);
	cout << "Succesfully finished Negative track propagation  -------------- with AN: " << trackStateM2.globalPosition() << endl;
      } catch (MagVolumeOutsideValidity & duh){
	cout <<  "MagVolumeOutsideValidity not properly caught!! Lost this track " << endl;
      }
    }
  }
  cout << " Succesfully reached the END of this test !!!!!!!!!! " << endl;
}


Surface::RotationType RKTest::rotation( const GlobalVector& zDir) const
{
  GlobalVector zAxis = zDir.unit();
  GlobalVector yAxis( zAxis.y(), -zAxis.x(), 0); 
  GlobalVector xAxis = yAxis.cross( zAxis);
  return Surface::RotationType( xAxis, yAxis, zAxis);
}


DEFINE_FWK_MODULE(RKTest);

