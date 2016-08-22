// author : Pierluigi Bortignon
// email : pierluigi.bortignon@gmail.com
// date : 16.10.2015 - Fermilab
// version : 1.0

#ifndef PhysicsTools_Heppy_ColorFlow_h
#define PhysicsTools_Heppy_ColorFlow_h

#include <vector>
#include <iostream>
#include <cmath>
#include <TVector2.h>
#include <TLorentzVector.h>
#include <TMath.h>

#include "DataFormats/Math/interface/LorentzVector.h"


namespace heppy{
class ColorFlow {

 // The ColorFlow class groups a set of variables that could be sensitive to the color flow connection between two jets.
 // The variable is inspired from this paper: http://arxiv.org/abs/1001.5027
 //
 // The class defines a 2D vector (a pull vector) per each jet. The pull vector represents the direction of the radiation in the rapiity-phi plane.
 // The interface suggests to use the phi of the pull vector (in the phi-rapidity plane) and its magnitude as sensitive variables to color connection.
    
public:

  ColorFlow(std::vector<math::XYZTLorentzVector> pfCands);
 
  TVector2 get_pull_vector() { return pull_vector_; };

  float get_pull_vector_phi() { return pull_vector_.Phi(); };

  float get_pull_vector_mag() { return pull_vector_.Mod(); };

private:

  void init();

  // It return the 4-momentum sum of pfCands.
  TLorentzVector CalculateJetDirection( std::vector<math::XYZTLorentzVector> pfCands_ );

  std::vector<math::XYZTLorentzVector> pfCands_;
  TVector2 pull_vector_;
  TLorentzVector pi_;
  TLorentzVector J_;
  TVector2 r_; 
  float r_mag_;
  float r_phi_;
  float pf_cand_pt_;
  unsigned int n_of_pf_cands_;
  
};
}
#endif   
 
