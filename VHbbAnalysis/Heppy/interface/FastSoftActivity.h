#ifndef PhysicsTools_Heppy_FastSoftActivity_h
#define PhysicsTools_Heppy_FastSoftActivity_h

#include <vector>
#include <iostream>
#include <cmath>
#include <TLorentzVector.h>
#include <TMath.h>
#include "DataFormats/Math/interface/LorentzVector.h"

#include <boost/shared_ptr.hpp>
#include <fastjet/internal/base.hh>
#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"
#include "fastjet/PseudoJet.hh"

namespace heppy{
class FastSoftActivity {
    
 public:
  typedef math::XYZTLorentzVector LorentzVector;

  FastSoftActivity(const std::vector<LorentzVector> & objects, double ktpower, double rparam,
		   const LorentzVector &j1,const LorentzVector &j2,double sumDeltaRMin);

  /// get grouping (inclusive jets)
  std::vector<LorentzVector> getGrouping(double ptMin = 0.0);

 private:
  // pack the returns in a fwlite-friendly way
  std::vector<LorentzVector> makeP4s(const std::vector<fastjet::PseudoJet> &jets) ;

  // used to handle the inputs
  std::vector<fastjet::PseudoJet> fjInputs_;        // fastjet inputs

  double ktpower_;
  double rparam_;
  LorentzVector j1_;
  LorentzVector j2_;
  double sumDeltaRMin_;
 
  /// fastjet outputs
  typedef boost::shared_ptr<fastjet::ClusterSequence>  ClusterSequencePtr;
  ClusterSequencePtr fjClusterSeq_;    
  std::vector<fastjet::PseudoJet> inclusiveJets_; 
  std::vector<fastjet::PseudoJet> exclusiveJets_; 
};
}
#endif   
 
