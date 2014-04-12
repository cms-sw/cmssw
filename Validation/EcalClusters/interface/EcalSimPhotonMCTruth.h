#ifndef EcalSimPhotonMCTruth_h
#define EcalSimPhotonMCTruth_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include <vector>

/** \class EcalSimPhotonMCTruth
 *       
 *  This class stores all the MC truth information needed about the
 *  conversion for containment correction
 *  original code: PhotonMCTruth (N.Marinelli)
 *
 */

class EcalSimPhotonMCTruth {
 public:
  EcalSimPhotonMCTruth() : isAConversion_(0),thePhoton_(0.,0.,0.,0.), theR_(0.), theZ_(0.), 
    theConvVertex_(0.,0.,0.,0.) {};
  
  EcalSimPhotonMCTruth(const math::XYZTLorentzVectorD& v) : thePhoton_(v) {};
  
  EcalSimPhotonMCTruth(int isAConversion,const math::XYZTLorentzVectorD& v, float rconv, float zconv,
		       const math::XYZTLorentzVectorD& convVertex, const math::XYZTLorentzVectorD& pV, const std::vector<const SimTrack *>& tracks );
  
  math::XYZTLorentzVectorD primaryVertex() const {return thePrimaryVertex_;}
  int isAConversion() const { return isAConversion_;}
  float radius() const {return theR_;}
  float z() const {return theZ_;}
  math::XYZTLorentzVectorD fourMomentum() const {return thePhoton_;}
  math::XYZTLorentzVectorD vertex() const {return theConvVertex_;}
  std::vector<const SimTrack *> simTracks() const {return tracks_;} 
  
 private:
  
  int isAConversion_;
  math::XYZTLorentzVectorD thePhoton_;
  float theR_;
  float theZ_;
  math::XYZTLorentzVectorD theConvVertex_;
  math::XYZTLorentzVectorD thePrimaryVertex_;
  std::vector<const SimTrack *> tracks_;

};

#endif

