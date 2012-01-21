#ifndef Match_h
#define Match_h

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Candidate/interface/Candidate.h"

class Match {
  
 public:

  Match(){};
  ~Match(){};
  double operator()(const reco::Candidate& ref, const reco::Candidate& rec)
  { return reco::deltaR(ref.eta(), ref.phi(), rec.eta(), rec.phi()); };
};

#endif
