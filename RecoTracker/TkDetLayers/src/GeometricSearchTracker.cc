#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

GeometricSearchTracker::GeometricSearchTracker(const vector<BarrelDetLayer*>& bl,
					       const vector<ForwardDetLayer*>& negFl,
					       const vector<ForwardDetLayer*>& posFl):
  theBarrelLayers(bl.begin(),bl.end()),
  theNegForwardLayers(negFl.begin(),negFl.end()),
  thePosForwardLayers(posFl.begin(),posFl.end())
{
  theForwardLayers.assign(negFl.begin(),negFl.end());
  theForwardLayers.insert(theForwardLayers.end(),posFl.begin(),posFl.end());
  theAllLayers.assign(bl.begin(),bl.end());
  theAllLayers.insert(theAllLayers.end(),
		      theForwardLayers.begin(),
		      theForwardLayers.end());
}


GeometricSearchTracker::~GeometricSearchTracker(){
  for(vector<DetLayer*>::const_iterator it=theAllLayers.begin(); it!=theAllLayers.end();it++){
    delete *it;
  }
  
}
