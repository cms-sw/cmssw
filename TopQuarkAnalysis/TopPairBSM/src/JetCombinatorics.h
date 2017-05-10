#ifndef JetCombinatorics_h
#define JetCombinatorics_h

/**_________________________________________________________________
   class:   JetCombinatorics.h
   package: 


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: JetCombinatorics.h,v 1.1 2012/10/11 14:26:45 eulisse Exp $

________________________________________________________________**/

#include "TLorentzVector.h"
#include "TString.h"
#include "TH1F.h"
#include "TFile.h"
#include "TMath.h"
#include <map>
#include <vector>
#include <iostream>

class Combo {


  public:

	Combo() {
		
	  MW = 84.2;//79.8;
	  Mtop_h = 180.7;//175.;
	  Mtop_l = 174.9;
	  sigmaHadW = 10.5;//2.*7.6;
	  sigmaHadt = 19.2;//2.*12.5;
	  sigmaLept = 24.2;//2.*15.6;

	  SumEt_ = 0.;
	  usebtag_ = false;
	  useMtop_ = true;

	  useFlv_ = false;
	  Wp_flv_ = Wq_flv_ = Hadb_flv_ = Lepb_flv_ = 1.;
	}
	~Combo(){};

	void SetWp(TLorentzVector Wp) { Wp_ = Wp; }
	void SetWq(TLorentzVector Wq) { Wq_ = Wq; }
	void SetHadb(TLorentzVector Hadb) { Hadb_ = Hadb; }
	void SetLepW(TLorentzVector LepW) { LepW_ = LepW; }
	void SetLepb(TLorentzVector Lepb) { Lepb_ = Lepb; }
	// flavor corrections
	void ApplyFlavorCorrections(bool option=true){ useFlv_ = option;}
	void SetFlvCorrWp( double corr ) { Wp_flv_ = corr; }
	void SetFlvCorrWq( double corr ) { Wq_flv_ = corr; }
	void SetFlvCorrHadb( double corr ) { Hadb_flv_ = corr; }
	void SetFlvCorrLepb( double corr ) { Lepb_flv_ = corr; }
	// b tagging
	void SetWp_disc(double disc) { Wp_disc_ = disc;}
	void SetWq_disc(double disc) { Wq_disc_= disc;}
	void SetHadb_disc(double disc) { Hadb_disc_= disc;}
	void SetLepb_disc(double disc) { Lepb_disc_= disc;}
	void SetbDiscPdf(TString filename) { 
	  pdffile_ = TFile::Open(filename);
	  hdisc_b_ = (TH1F*) gDirectory->Get("hdiscNorm_b");
	  hdisc_cl_ = (TH1F*) gDirectory->Get("hdiscNorm_cl");
	}
	void SetSigmas(int type=0) {

	  // type == 0 take defaults
	  if (type==1) {
	    // JES +10%
	    MW = 87.2;
	    Mtop_h = 193.2;
	    Mtop_l = 179.0;
	    sigmaHadW = 13.0;
	    sigmaHadt = 22.8;
	    sigmaLept = 26.3;
	  }
	  if (type==-1) {
            // JES -10%
            MW = 81.6;
            Mtop_h = 169.3;
            Mtop_l = 171.4;
            sigmaHadW =8.9;
            sigmaHadt =17.9;
            sigmaLept =22.6;
	  }

	}
	void Usebtagging(bool option = true) { usebtag_ = option;}
	void SetMinMassLepW( double mass ) { minMassLepW_ = mass; }
	void SetMaxMassLepW( double mass ) { maxMassLepW_ = mass; }
	void SetMinMassHadW( double mass ) { minMassHadW_ = mass; }
	void SetMaxMassHadW( double mass ) { maxMassHadW_ = mass; }
	void SetMinMassLepTop( double mass ) { minMassLepTop_ = mass; }
	void SetMaxMassLepTop( double mass ) { maxMassLepTop_ = mass; }
	void UseMtopConstraint(bool option=true) { useMtop_ = option; }
		
	void analyze() {

		if ( useFlv_ ) {
			Wp_ = Wp_flv_ * Wp_;
			Wq_ = Wq_flv_ * Wq_;
			Hadb_ = Hadb_flv_ * Hadb_;
			Lepb_ = Lepb_flv_ * Lepb_;
		}

		HadW_ = Wp_ + Wq_;
		HadTop_ = HadW_ + Hadb_;
		LepTop_ = LepW_ + Lepb_;
		TopPair_ = HadTop_ + LepTop_;

		//double sigmaHadW = 10.5;//2.*7.6;
		//double sigmaHadt = 19.2;//2.*12.5;
		//double sigmaLept = 24.2;//2.*15.6;
		
		double chiHadW = (HadW_.M() - MW)/sigmaHadW;
		double chiHadt = (HadTop_.M() - Mtop_h)/sigmaHadt;
		double chiLept = (LepTop_.M() - Mtop_l)/sigmaLept;

		if ( useMtop_ ) {
			chi2_ = chiHadW*chiHadW + chiHadt*chiHadt + chiLept*chiLept;
			Ndof_ = 3;
		} else {
			chi2_ = chiHadW*chiHadW + (HadTop_.M() - LepTop_.M())*(HadTop_.M() - LepTop_.M())/(sigmaHadt*sigmaHadt+sigmaLept*sigmaLept);
			Ndof_ = 2;
		}
		
		SumEt_ = HadTop_.Pt();

		if ( usebtag_ ) {

			double gauss_norm = (2.)*TMath::Log(sigmaHadW*TMath::Sqrt(2*TMath::Pi())) +
				(2.)*TMath::Log(sigmaHadt*TMath::Sqrt(2*TMath::Pi())) + (2.)*TMath::Log(sigmaLept*TMath::Sqrt(2*TMath::Pi()));

			double LR_Wp; double LR_Wq;
			double LR_Hadb; double LR_Lepb;

			double LR_den = 0;
			LR_den = ( getPdfValue("cl", Wp_disc_) + getPdfValue("b", Wp_disc_));
			if (LR_den == 0 ) LR_Wp = 1e-5;
			else LR_Wp = getPdfValue( "cl", Wp_disc_ )/ LR_den;

			LR_den = ( getPdfValue("cl", Wq_disc_) + getPdfValue("b", Wq_disc_));
			if (LR_den == 0 ) LR_Wq = 1e-5;
			else LR_Wq = getPdfValue( "cl", Wq_disc_ )/ LR_den;

			LR_den = ( getPdfValue("cl", Hadb_disc_) + getPdfValue("b", Hadb_disc_));
			if (LR_den == 0 ) LR_Hadb = 1e-5;
			else LR_Hadb = getPdfValue( "b", Hadb_disc_ )/ LR_den;

			LR_den = ( getPdfValue("cl", Lepb_disc_) + getPdfValue("b", Lepb_disc_));
			if (LR_den == 0 ) LR_Lepb = 1e-5;
			else LR_Lepb = getPdfValue( "b", Lepb_disc_ )/ LR_den;

			double btag_norm = (-0.25-TMath::Log(4)/2);
			double btag_N2LL = btag_norm*4.*( LR_Wp * TMath::Log(LR_Wp/4) + LR_Wq*TMath::Log(LR_Wq/4) + LR_Hadb*TMath::Log(LR_Hadb/4) + LR_Lepb*TMath::Log(LR_Lepb/4) );
		  
			chi2_ += btag_N2LL + gauss_norm;
			Ndof_ += 3;
			pdffile_->Close();
		}
	}

	TLorentzVector GetWp() { return Wp_; }
	TLorentzVector GetWq() { return Wq_; }
	TLorentzVector GetHadW() { return HadW_; }
	TLorentzVector GetLepW() { return LepW_; }
	TLorentzVector GetHadb() { return Hadb_; }
	TLorentzVector GetLepb() { return Lepb_; }
	TLorentzVector GetHadTop() { return HadTop_; }
	TLorentzVector GetLepTop() { return LepTop_; }
	TLorentzVector GetTopPair() { return TopPair_; }
	double GetChi2() { return chi2_; }
	double GetNdof() { return Ndof_; }
	double GetSumEt() { return SumEt_; }
	int GetIdHadb() { return IdHadb_;}
	int GetIdWp() { return IdWp_; }
	int GetIdWq() { return IdWq_; }
	int GetIdLepb() { return IdLepb_;}
	void SetIdHadb(int id) { IdHadb_ = id;}
	void SetIdWp(int id) { IdWp_ = id; }
	void SetIdWq(int id) { IdWq_ = id; }
	void SetIdLepb(int id) { IdLepb_ = id;}
	void Print() {
	  std::cout << " jet Wp  : px = " << Wp_.Px() << " py = " <<  Wp_.Py() << " pz = " << Wp_.Pz() << " e = " << Wp_.E() << std::endl;
	  std::cout << " jet Wq  : px = " << Wq_.Px() << " py = " <<  Wq_.Py() << " pz = " << Wq_.Pz() << " e = "<< Wq_.E() << std::endl;
	  std::cout << " jet Hadb: px = " << Hadb_.Px() << " py = " <<  Hadb_.Py() <<" pz = " << Hadb_.Pz() <<" e = "<< Hadb_.E() << std::endl;
	  std::cout << " jet Lepb: px = " << Lepb_.Px() << " py = " <<  Lepb_.Py() <<" pz = " << Lepb_.Pz() <<" e = "<< Lepb_.E() << std::endl;
	  std::cout << " chi-squared = " << chi2_ << " sumEt = " << SumEt_ << std::endl;
	}
	double getPdfValue(std::string flavor, double disc) {
	  double pdf= 0;
	  TH1F *hpdf;
	  if ( flavor == "b" ) hpdf = hdisc_b_;
	  else hpdf = hdisc_cl_;
	  int bin = hpdf->GetXaxis()->FindBin( disc );
	  pdf = hpdf->GetBinContent( bin );
	  if ( disc < -10 || disc >50 ) return 0;
	  //if ( pdf == 0 ) return 1.e-7;
	  return pdf;
	}
	
  private:
	
	TLorentzVector Wp_;
	TLorentzVector Wq_;
	TLorentzVector HadW_;
	TLorentzVector Hadb_;
	TLorentzVector HadTop_;
	TLorentzVector LepW_;
	TLorentzVector Lepb_;	
	TLorentzVector LepTop_;
	TLorentzVector TopPair_;
	
	bool usebtag_;
	bool useMtop_;
	double Wp_disc_;
	double Wq_disc_;
	double Hadb_disc_;
	double Lepb_disc_;
	TFile *pdffile_;
	TH1F *hdisc_b_;
	TH1F *hdisc_cl_;

	double Wp_flv_, Wq_flv_, Hadb_flv_, Lepb_flv_;
	bool useFlv_;
	double chi2_;
	double Ndof_;
	double SumEt_;
	double minMassLepW_;
	double maxMassLepW_;
	double minMassHadW_;
	double maxMassHadW_;
	
	double minMassLepTop_;
	double maxMassLepTop_;

	double MW;
	double Mtop_h;
	double Mtop_l;
	double sigmaHadW;
	double sigmaHadt;
	double sigmaLept;


	int IdHadb_;
	int IdWp_;
	int IdWq_;
	int IdLepb_;
	
};

struct minChi2
{
  bool operator()(Combo s1, Combo s2) const
  {
    return s1.GetChi2() <= s2.GetChi2();
  }
};

struct maxSumEt
{
  bool operator()(Combo s1, Combo s2) const
  {
    return s1.GetSumEt() >= s2.GetSumEt();
  }
};


class JetCombinatorics {

  public:

	JetCombinatorics();
	~JetCombinatorics();

	void Verbose() {
	  verbosef = true;
	}

	std::map< int, std::string > Combinatorics(int k, int max = 6);
	std::map< int, std::string > NestedCombinatorics();

	void FourJetsCombinations(std::vector<TLorentzVector> jets, std::vector<double> bdiscriminators );
	void SetFlavorCorrections(std::vector<double > vector ) { flavorCorrections_ = vector; }
	void SetMaxNJets(int n) { maxNJets_ = n; }
	Combo GetCombination(int n=0);
	Combo GetCombinationSumEt(int n=0);
	int GetNumberOfCombos() { return ( (int)allCombos_.size() ); } 
	//void SetCandidate( std::vector< TLorentzVector > JetCandidates );

	void SetSigmas(int type = 0) {
	  SigmasTypef = type;
	}
	void SetLeptonicW( TLorentzVector LepW ) { theLepW_ = LepW; }

	void SetMinMassLepW( double mass ) { minMassLepW_ = mass; }
	void SetMaxMassLepW( double mass ) { maxMassLepW_ = mass; }
	void SetMinMassHadW( double mass ) { minMassHadW_ = mass; }
	void SetMaxMassHadW( double mass ) { maxMassHadW_ = mass; }
	void SetMinMassLepTop( double mass ) { minMassLepTop_ = mass; }
	void SetMaxMassLepTop( double mass ) { maxMassLepTop_ = mass; }

	void UsebTagging( bool option = true ) { UsebTagging_ = option; }
	void ApplyFlavorCorrection( bool option = true ) { UseFlv_ = option; }
	void UseMtopConstraint( bool option = true) { UseMtop_ = option; }
	void SetbTagPdf( TString name ) { bTagPdffilename_ = name; }
	void Clear();

	std::vector< TLorentzVector > TwoCombos();
	std::vector< TLorentzVector > ThreeCombos();

	void RemoveDuplicates( bool option) { removeDuplicates_ = option; }

	std::vector< TLorentzVector > GetComposites();
	void AnalyzeCombos();


  private:

	//int kcombos_;
	//int maxcombos_;
	int SigmasTypef;
	bool verbosef;
	std::map< int, std::string > Template4jCombos_;
	std::map< int, std::string > Template5jCombos_;
	std::map< int, std::string > Template6jCombos_;
	std::map< int, std::string > Template7jCombos_;

	std::vector< double > flavorCorrections_;
	int maxNJets_;
	bool UsebTagging_;
	bool UseMtop_;
	TString bTagPdffilename_;
	bool UseFlv_;
	
	TLorentzVector theLepW_;

	double minMassLepW_;
	double maxMassLepW_;
	double minMassHadW_;
	double maxMassHadW_;
	double minMassLepTop_;
	double maxMassLepTop_;
	
	std::map< Combo, int, minChi2 > allCombos_;
	std::map< Combo, int, maxSumEt > allCombosSumEt_;

	Double_t minPhi_;
	double chi2_;
	int ndf_;
	bool removeDuplicates_;
	
	std::vector< TLorentzVector > cand1_;
	std::vector< TLorentzVector > cand2_;
	std::vector< TLorentzVector > cand3_;

	//int nLists_;
	
	//std::vector< TLorentzVector > composites_;
	
};

#endif
