#ifndef Balance_h
#define Balance_h

#include "TMath.h"
#include "DataFormats/Math/interface/deltaR.h"

class Balance {
  
 public:

  Balance(){};
  ~Balance(){};
  double operator()(const reco::Candidate& ref, const reco::Candidate& rec)
  { return fabs( fabs(reco::deltaPhi(ref.phi(), rec.phi()))-TMath::Pi()); };
};

#endif
