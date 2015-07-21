#ifndef L1EGammaCrystalsCluster_h
#define L1EGammaCrystalsCluster_h

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace l1slhc
{

  class L1EGCrystalCluster : public reco::LeafCandidate {
    public:
      L1EGCrystalCluster() : LeafCandidate(), hovere_(0.), iso_(0.), PUcorrPt_(0.), bremStrength_(0.) {};
      L1EGCrystalCluster(const PolarLorentzVector& p4, float hovere, float iso, DetId seedCrystal, float PUcorrPt = 0., float bremStrength = 0.) : LeafCandidate(0, p4), hovere_(hovere), iso_(iso), seedCrystal_(seedCrystal), PUcorrPt_(PUcorrPt), bremStrength_(bremStrength) {};
      L1EGCrystalCluster(const LorentzVector& p4, float hovere, float iso, float PUcorrPt = 0., float bremStrength = 0.) : LeafCandidate(0, p4), hovere_(hovere), iso_(iso), PUcorrPt_(PUcorrPt), bremStrength_(bremStrength) {};
      virtual ~L1EGCrystalCluster() {};
      inline float hovere() const { return hovere_; };
      inline float isolation() const { return iso_; };
      inline float PUcorrPt() const { return PUcorrPt_; };
      inline float bremStrength() const { return bremStrength_; };
      inline DetId seedCrystal() const { return seedCrystal_; };
      void SetCrystalPtInfo(std::vector<float> &info) {
         std::sort(info.begin(), info.end());
         std::reverse(info.begin(), info.end());
         crystalPt_ = info;
      };
      void SetExperimentalParams(const std::map<std::string, float> &params) { experimentalParams_ = params; };
      const std::map<std::string, float> GetExperimentalParams() const { return experimentalParams_; };
      inline float GetExperimentalParam(std::string name) const { return experimentalParams_.at(name); };

      // The index range depends on the algorithm eta,phi window, currently 3x5
      // The pt should always be ordered.
      inline float GetCrystalPt(unsigned int index) const { return (index < crystalPt_.size()) ? crystalPt_[index] : 0.; };
    
    private:
      // HCal energy in region behind cluster (for size, look in producer) / ECal energy in cluster
      float hovere_;
      // ECal isolation (for outer window size, again look in producer)
      float iso_;
      // DetId of seed crystal used to make cluster (could be EBDetId or EEDetId)
      DetId seedCrystal_;
      // Pileup-corrected energy deposit, not studied carefully yet, don't use
      float PUcorrPt_;
      // Bremstrahlung strength, should be proportional to the likelihood of a brem.
      float bremStrength_;
      // Crystal pt (in order of strength) for all crystals in the cluster
      std::vector<float> crystalPt_;
      // For investigating novel algorithm parameters
      std::map<std::string, float> experimentalParams_;
  };
  
  
  // Concrete collection of output objects (with extra tuning information)
  typedef std::vector<l1slhc::L1EGCrystalCluster> L1EGCrystalClusterCollection;
}
#endif

