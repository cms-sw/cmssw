#ifndef L1EGammaCrystalsCluster_h
#define L1EGammaCrystalsCluster_h

#include <vector>
#include "DataFormats/Candidate/interface/LeafCandidate.h"

namespace l1slhc
{

  class L1EGCrystalCluster : public reco::LeafCandidate {
    public:
      L1EGCrystalCluster() : LeafCandidate(), hovere_(0.), iso_(0.), PUcorrPt_(0.) {};
      L1EGCrystalCluster(const PolarLorentzVector& p4, float hovere, float iso, float PUcorrPt = 0.) : LeafCandidate(0, p4), hovere_(hovere), iso_(iso), PUcorrPt_(PUcorrPt) {};
      L1EGCrystalCluster(const LorentzVector& p4, float hovere, float iso, float PUcorrPt = 0.) : LeafCandidate(0, p4), hovere_(hovere), iso_(iso), PUcorrPt_(PUcorrPt) {};
      virtual ~L1EGCrystalCluster() {};
      inline float hovere() const { return hovere_; };
      inline float isolation() const { return iso_; };
      inline float PUcorrPt() const { return PUcorrPt_; };
    
    private:
      // HCal energy in region behind cluster (for size, look in producer) / ECal energy in cluster
      float hovere_;
      // ECal isolation (for outer window size, again look in producer)
      float iso_;
      // Pileup-corrected energy deposit, not studied carefully yet, don't use
      float PUcorrPt_;
  };
  
  
  // Concrete collection of output objects (with extra tuning information)
  typedef std::vector<l1slhc::L1EGCrystalCluster> L1EGCrystalClusterCollection;
}
#endif

