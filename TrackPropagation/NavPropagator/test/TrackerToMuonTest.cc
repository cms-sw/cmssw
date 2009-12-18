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

#include "FWCore/MessageLogger/interface/MessageLogger.h"


////////////////////////////////////////////////////////////////////////////

#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"

#include  "TrackPropagation/NavPropagator/interface/NavPropagator.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

#include <map>
using namespace std;

class TrackerToMuonTest : public edm::EDAnalyzer {
public:
  TrackerToMuonTest(const edm::ParameterSet& pset) {}

  ~TrackerToMuonTest(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){
    using namespace edm;
    ESHandle<MagneticField> magfield;
    setup.get<IdealMagneticFieldRecord>().get(magfield);
    try{
    const GlobalPoint g(0,0,0);
    std::cout << "B-field(T) at (0,0,0)(cm): " << magfield->inTesla(g) << std::endl;
    } catch (MagVolumeOutsideValidity & duh) { 
      std::cout << " Caught MagVolumeOutsideValidity exception before we even started !!!" << std::endl;
    }
    propagateToMuon( &(*magfield));
  }

private:

  typedef TrajectoryStateOnSurface                   TSOS;

  void propagateToMuon( const MagneticField* field) const;

  Surface::RotationType rotation( const GlobalVector& zAxis) const;

};

void TrackerToMuonTest::propagateToMuon( const MagneticField* field) const
{

  NavPropagator prop(field);

  for (float phi = -0.05; phi<0.25*3.1419 ; phi+=0.1) {
    for (float costh = 0.25; costh<0.35 ; costh+=0.05) {
      cout << "And now trying costh, phi = " << costh << ", " << phi << endl;
      
      PlaneBuilder pb;
      float sinth = sqrt(1-costh*costh);
      GlobalVector startingMomentum(50*sin(phi)*sinth,50*cos(phi)*sinth,-50*costh); // locks up with 20
      GlobalPoint startingPosition(-3,-2,-1);
      Surface::RotationType rot = rotation( startingMomentum);

      PlaneBuilder::ReturnType trackerPlane = pb.plane( startingPosition, rot);

      AlgebraicSymMatrix C(5,1);
      C *= 0.01;
      CurvilinearTrajectoryError err(C);

      TSOS startingStateP( GlobalTrajectoryParameters(startingPosition, 
						      startingMomentum, 1, field), 
			   err, *trackerPlane);

      TSOS startingStateM( GlobalTrajectoryParameters(startingPosition, 
						      startingMomentum, -1, field), 
			   err, *trackerPlane);

      float propDistance = 900; // 10 meters in [cm]
      GlobalPoint targetPos( (propDistance*startingMomentum.unit()).basicVector());
      PlaneBuilder::ReturnType muonPlane = pb.plane( targetPos, rot);

      try {
	TSOS muonStateP = prop.propagate( startingStateP, *muonPlane);
	cout << "Succesfully finished Positive muon propagation  -------------- at position: " << muonStateP.globalPosition() << endl;
      } catch (MagVolumeOutsideValidity & duh){
	cout << "MagVolumeOutsideValidity not properly caught!! Lost this muon " << endl;
      }
      // cout << "Positive muon ended at ==== " << muonStateP.globalPosition() << endl;

      try {
	TSOS muonStateM = prop.propagate( startingStateM, *muonPlane);
	cout << "Succesfully finished Negative muon propagation  -------------- at position: " << muonStateM.globalPosition() << endl;
      } catch (MagVolumeOutsideValidity & duh){
	cout <<  "MagVolumeOutsideValidity not properly caught!! Lost this muon " << endl;
      }
      // cout << "Negative muon ended at ==== " << muonStateP.globalPosition() << endl;

    }
  }
  cout << "And Yes, succesfully reached the END of this test !!!!!!!!!! " << endl;
}


Surface::RotationType TrackerToMuonTest::rotation( const GlobalVector& zDir) const
{
  GlobalVector zAxis = zDir.unit();
  GlobalVector yAxis( zAxis.y(), -zAxis.x(), 0); 
  GlobalVector xAxis = yAxis.cross( zAxis);
  return Surface::RotationType( xAxis, yAxis, zAxis);
}


DEFINE_FWK_MODULE(TrackerToMuonTest);

