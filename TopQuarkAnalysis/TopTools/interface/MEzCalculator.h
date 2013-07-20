#ifndef TopTools_MEzCalculator_h
#define TopTools_MEzCalculator_h

/**_________________________________________________________________
   class:   MEzCalculator.h

 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: MEzCalculator.h,v 1.5 2013/05/30 20:47:31 gartung Exp $

________________________________________________________________**/

#include "DataFormats/PatCandidates/interface/Particle.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "TLorentzVector.h"

class MEzCalculator {
  
 public:
  /// constructor
  MEzCalculator();
  /// destructor
  ~MEzCalculator();
  /// Set MET
  void SetMET(const pat::MET& MET) { MET_ = MET; } ;
  void SetMET(const TLorentzVector& MET) {
	  pat::Particle::LorentzVector p(MET.Px(),MET.Py(),MET.Pz(),MET.E());
	  MET_.setP4(p);
  }
  /// Set lepton
  void SetLepton(const pat::Particle& lepton, bool isMuon = true) {
	  lepton_ = lepton;
	  isMuon_ = isMuon;
  };
  void SetLepton(const TLorentzVector& lepton) {
	  pat::Particle::LorentzVector p(lepton.Px(), lepton.Py(), lepton.Pz(), lepton.E() );
	  lepton_.setP4(p);
  }
  /// Calculate MEz
  /// options to choose roots from quadratic equation:
  /// type = 0 : if real roots, pick the one nearest to
  ///                     the lepton Pz except when the Pz so chosen
  ///                     is greater than 300 GeV in which case pick
  ///                     the most central root.
  /// type = 1 (default): if real roots, choose the one closest to the lepton Pz
  ///           if complex roots, use only the real part.
  /// type = 2: if real roots, choose the most central solution.
  ///           if complex roots, use only the real part.
  double Calculate(int type = 1);
  /// check for complex root
  bool IsComplex() const { return isComplex_; };
  /// verbose
  void Print() {
	  std::cout << " METzCalculator: pxmu = " << lepton_.px() << " pzmu= " << lepton_.pz() << std::endl;
	  std::cout << " METzCalculator: pxnu = " << MET_.px() << " pynu= " << MET_.py() << std::endl;
  }

 private:
  
  bool isComplex_;
  pat::Particle lepton_;
  pat::MET MET_;
  bool isMuon_;
};

#endif
