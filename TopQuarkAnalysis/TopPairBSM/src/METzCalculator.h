#ifndef METzCalculator_h
#define METzCalculator_h

/**_________________________________________________________________
   class:   METzCalculator.h

 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: METzCalculator.h,v 1.1 2012/10/11 14:26:45 eulisse Exp $

________________________________________________________________**/

//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Particle.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "TLorentzVector.h"

class METzCalculator {

  public:
	/// constructor
	METzCalculator();
		//METzCalculator(const edm::ParameterSEt& iConf);
	/// destructor
	virtual ~METzCalculator();
	/// Set MET
	void SetMET(const pat::MET &MET) { MET_ = MET; } ;
	void SetMET(TLorentzVector MET) {
		pat::Particle::LorentzVector p(MET.Px(),MET.Py(),MET.Pz(),MET.E());
		MET_.setP4(p);
	}
    /// Set Muon
	void SetMuon(const pat::Particle &lepton) { lepton_ = lepton; };
	void SetMuon(TLorentzVector lepton) {
		pat::Particle::LorentzVector p(lepton.Px(), lepton.Py(), lepton.Pz(), lepton.E() );
		lepton_.setP4(p);
	}
    /// Calculate MEz
	/// options to choose roots from quadratic equation:
	/// type = 0 (defalut): if real roots, pick the one nearest to
	///                     the lepton Pz except when the Pz so chosen
	///                     is greater than 300 GeV in which case pick
	///                     the most central root.
	/// type = 1: if real roots, choose the one closest to the lepton Pz
	///           if complex roots, use only the real part.
	/// type = 2: if real roots, choose the most central solution.
	///           if complex roots, use only the real part.
	/// type = 3: if real roots, pick the largest value of the cosine*
	double Calculate(int type = 0);
    /// check for complex root
	bool IsComplex() const { return isComplex_; };

	void Print() {
		std::cout << " METzCalculator: pxmu = " << lepton_.px() << " pzmu= " << lepton_.pz() << std::endl;
		std::cout << " METzCalculator: pxnu = " << MET_.px() << " pynu= " << MET_.py() << std::endl;
	}
	
  private:

	bool isComplex_;
	pat::Particle lepton_;
	pat::MET MET_;
	
};

#endif
