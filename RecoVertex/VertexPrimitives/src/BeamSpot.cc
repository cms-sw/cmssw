#include "RecoVertex/VertexPrimitives/interface/BeamSpot.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"

BeamSpot::BeamSpot() : thePos(0, 0, 0), 
		       theErr(0.0015*0.0015, 0., 0.0015*0.0015, 
			      0., 0., 5.3*5.3) 
{}

BeamSpot::BeamSpot(const reco::BeamSpot & beamSpot) :
	 thePos(RecoVertex::convertPos(beamSpot.position())),
	 theErr(RecoVertex::convertError(beamSpot.covariance3D()))
{}

BeamSpot::BeamSpot(const BeamSpotObjects * beamSpot)
{
  thePos=GlobalPoint(beamSpot->GetX(), beamSpot->GetY(), beamSpot->GetZ());
  double dx=sqrt(pow(beamSpot->GetBeamWidth(),2)+pow(beamSpot->GetXError(),2));
  double dy=sqrt(pow(beamSpot->GetBeamWidth(),2)+pow(beamSpot->GetYError(),2));
  double dz=beamSpot->GetSigmaZ();
  theErr=GlobalError(dx*dx,0.,dy*dy, 0., 0., dz*dz);
}

  // to be replaced by ParameterSet
  /*
  ConfigurableVector<float> conf_pos ("BeamSpot:position");
  if (conf_pos.size() == 3) {
    thePos = GlobalPoint(conf_pos[0], conf_pos[1], conf_pos[2]);
  }

  ConfigurableVector<float> conf_spread ("BeamSpot:spread");
  if (conf_spread.size() == 3) {
    theErr = GlobalError(conf_spread[0]*conf_spread[0], 
			 0., 
			 conf_spread[1]*conf_spread[1], 
			 0., 
			 0.,
			 conf_spread[2]*conf_spread[2]);
  }
  */
