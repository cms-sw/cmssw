#include "Validation/EcalClusters/interface/EcalSimPhotonMCTruth.h"

#include <iostream>

EcalSimPhotonMCTruth::EcalSimPhotonMCTruth(int isAConversion,math::XYZTLorentzVectorD v, float rconv, float zconv,
					   math::XYZTLorentzVectorD convVertex,  
					   math::XYZTLorentzVectorD pV,  std::vector<const SimTrack *> tracks  ) :
  isAConversion_(isAConversion),
  thePhoton_(v), theR_(rconv), theZ_(zconv), theConvVertex_(convVertex), 
  thePrimaryVertex_(pV), tracks_(tracks)  {
}




