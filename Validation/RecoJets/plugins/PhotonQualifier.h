#ifndef PhotonQualifier_h
#define PhotonQualifier_h

#include <memory>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"


class PhotonQualifier {
  
 public:
  PhotonQualifier(const edm::ParameterSet&);
  ~PhotonQualifier(){};
  bool operator()(const reco::Photon&);

 private:

};

inline 
PhotonQualifier::PhotonQualifier(const edm::ParameterSet& cfg)
{
}

inline bool
PhotonQualifier::operator()(const reco::Photon& phot)
{
  return true;
}

#endif
