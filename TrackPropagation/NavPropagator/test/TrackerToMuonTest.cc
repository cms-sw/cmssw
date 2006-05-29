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

////////////////////////////////////////////////////////////////////////////

#include  "TrackPropagation/NavPropagator/interface/NavPropagator.h"
#include "Geometry/Surface/interface/PlaneBuilder.h"

#include <map>
using namespace std;


typedef 

class TrackerToMuonTest : public edm::EDAnalyzer {
public:
  TrackerToMuonTest(const edm::ParameterSet& pset) {}

  ~TrackerToMuonTest(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){
    using namespace edm;
    ESHandle<MagneticField> magfield;
    setup.get<IdealMagneticFieldRecord>().get(magfield);
    const GlobalPoint g(0.,0.,0.);
    std::cout << "B-field(T) at (0,0,0)(cm): " << magfield->inTesla(g) << std::endl;
    propagateToMuon( &(*magfield));
  }

private:

  typedef TrajectoryStateOnSurface                   TSOS;

  void propagateToMuon( const MagneticField* field) const;

  Surface::RotationType rotation( const GlobalVector& zAxis) const;

};

void TrackerToMuonTest::propagateToMuon( const MagneticField* field) const
{
  for (float phi = 0.; phi<3.15 ; phi+=0.1) {

  PlaneBuilder pb;

  GlobalVector startingMomentum(20*sin(phi),20*cos(phi),-2);
  GlobalPoint startingPosition( 0,0,0);
  Surface::RotationType rot = rotation( startingMomentum);

  PlaneBuilder::ReturnType trackerPlane = pb.plane( startingPosition, rot);

  TSOS startingStateP( GlobalTrajectoryParameters(startingPosition, 
						 startingMomentum, 1, field), 
		       *trackerPlane);

  TSOS startingStateM( GlobalTrajectoryParameters(startingPosition, 
						 startingMomentum, -1, field), 
		       *trackerPlane);

  float propDistance = 700; // 10 meters in [cm]
  GlobalPoint targetPos( (propDistance*startingMomentum.unit()).basicVector());
  PlaneBuilder::ReturnType muonPlane = pb.plane( targetPos, rot);

  NavPropagator prop(field);


    TSOS muonStateP = prop.propagate( startingStateP, *muonPlane);
    //  cout << "Positive muonState ==== " << muonStateP.globalPosition() << endl;
    
    cout << "Here is OK --------------------------" << endl;
    TSOS muonStateM = prop.propagate( startingStateM, *muonPlane);
    //cout << "Negative muonState " << muonStateM << endl;
    cout << "And Here is OK  --------------------------" << endl;
    }
}



Surface::RotationType TrackerToMuonTest::rotation( const GlobalVector& zDir) const
{
  GlobalVector zAxis = zDir.unit();
  GlobalVector yAxis( zAxis.y(), -zAxis.x(), 0); 
  GlobalVector xAxis = yAxis.cross( zAxis);
  return Surface::RotationType( xAxis, yAxis, zAxis);
}


DEFINE_FWK_MODULE(TrackerToMuonTest)

