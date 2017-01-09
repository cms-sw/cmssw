// author : Pierluigi Bortignon
// email : pierluigi.bortignon@gmail.com
// date : 16.10.2015 - Fermilab
// version : 0

//#include "PhysicsTools/Heppy/interface/ReclusterJets.h"
#include "VHbbAnalysis/Heppy/interface/ColorFlow.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

using namespace std;

namespace heppy{


  ColorFlow::ColorFlow(std::vector<math::XYZTLorentzVector> pfCands) 
    : pfCands_ (pfCands)
    {
      // initalisation of variables
      init();

      // if there are no more than 2 consituents there is no way to calculate the pull Angle - maybe in the future this can be revised.
      n_of_pf_cands_= pfCands_.size();
      if (n_of_pf_cands_ < 2) {
        return;
      }
      else { // calculate the pull_vector

        // recalculate jet direction using only pfCands_ 
        J_ = CalculateJetDirection(pfCands_);
        
        // calculate the pull vector. detailed explaination of the definition can be found in the paper reference in the header file.
        for ( unsigned int i=0; i < n_of_pf_cands_; i++){
          math::XYZTLorentzVector &pfCand = pfCands_.at(i);
          pf_cand_pt_ = pfCand.Pt();
          pi_.SetPtEtaPhiE(pfCand.Pt(),pfCand.Eta(),pfCand.Phi(),pfCand.E());
          r_.Set( pi_.Rapidity() - J_.Rapidity(), deltaPhi( pfCand.Phi(), J_.Phi() ) );
          r_mag_ = r_.Mod();
          pull_vector_ += ( pf_cand_pt_ / J_.Pt() ) * r_mag_ * r_;
        } // candidate loop
        return;
      } // else
    } // constructor

  void ColorFlow::init(){
      // initialisation of variables      
      pull_vector_.Set(0.,0.);
      J_.SetPtEtaPhiE(0.,0.,0.,0.);
  }

  TLorentzVector ColorFlow::CalculateJetDirection( std::vector<math::XYZTLorentzVector> pfCands_ ){
    for( unsigned int i = 0; i < n_of_pf_cands_ ; i++ ){
      math::XYZTLorentzVector &pfCand = pfCands_.at(i);
      pi_.SetPtEtaPhiE( pfCand.Pt(), pfCand.Eta(), pfCand.Phi(), pfCand.E() );
      J_+=pi_;
    }
    return J_;
  }

} // namespace heppy

