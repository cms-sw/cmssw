#include "Validation/EcalClusters/interface/EcalSimPhotonMCTruth.h"

#include <iostream>

EcalSimPhotonMCTruth::EcalSimPhotonMCTruth(int isAConversion,const math::XYZTLorentzVectorD& v, float rconv, float zconv,
					   const math::XYZTLorentzVectorD& convVertex,  
					   const math::XYZTLorentzVectorD& pV,  const std::vector<const SimTrack *>& tracks  ) :
  isAConversion_(isAConversion),
  thePhoton_(v), theR_(rconv), theZ_(zconv), theConvVertex_(convVertex), 
  thePrimaryVertex_(pV), tracks_(tracks)  {
}




