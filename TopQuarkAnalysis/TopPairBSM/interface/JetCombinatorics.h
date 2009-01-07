#ifndef JetCombinatorics_h
#define JetCombinatorics_h

/**_________________________________________________________________
   class:   JetCombinatorics.h
   package: 


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: JetCombinatorics.h,v 1.1.4.2 2009/01/07 22:31:00 yumiceva Exp $

________________________________________________________________**/

#include "TLorentzVector.h"
#include <map>
#include <vector>

class Combo {


  public:

	Combo() {
		
		MW = 79.8;
		Mtop = 175.;
		SumEt_ = 0.;
	}
	~Combo(){};

	void SetWp(TLorentzVector Wp) { Wp_ = Wp; }
	void SetWq(TLorentzVector Wq) { Wq_ = Wq; }
	void SetHadb(TLorentzVector Hadb) { Hadb_ = Hadb; }
	void SetLepW(TLorentzVector LepW) { LepW_ = LepW; }
	void SetLepb(TLorentzVector Lepb) { Lepb_ = Lepb; }
	void SetMinMassLepW( double mass ) { minMassLepW_ = mass; }
	void SetMaxMassLepW( double mass ) { maxMassLepW_ = mass; }
	void SetMinMassHadW( double mass ) { minMassHadW_ = mass; }
	void SetMaxMassHadW( double mass ) { maxMassHadW_ = mass; }
	void SetMinMassLepTop( double mass ) { minMassLepTop_ = mass; }
	void SetMaxMassLepTop( double mass ) { maxMassLepTop_ = mass; }
	
	void analyze() {

		HadW_ = Wp_ + Wq_;
		HadTop_ = HadW_ + Hadb_;
		LepTop_ = LepW_ + Lepb_;
		TopPair_ = HadTop_ + LepTop_;

		double sigmaHadW = 2.*7.6;
		double sigmaHadt = 2.*12.5;
		double sigmaLept = 2.*15.6;
		
		double chiHadW = (HadW_.M() - MW)/sigmaHadW;
		double chiHadt = (HadTop_.M() - Mtop)/sigmaHadt;
		double chiLept = (LepTop_.M() - Mtop)/sigmaLept;

		chi2_ = chiHadW*chiHadW + chiHadt*chiHadt + chiLept*chiLept;

		SumEt_ = HadTop_.Et();
		
	}
	
	TLorentzVector GetHadW() { return HadW_; }
	TLorentzVector GetLepW() { return LepW_; }
	TLorentzVector GetHadb() { return Hadb_; }
	TLorentzVector GetLepb() { return Lepb_; }
	TLorentzVector GetHadTop() { return HadTop_; }
	TLorentzVector GetLepTop() { return LepTop_; }
	TLorentzVector GetTopPair() { return TopPair_; }
	double GetChi2() { return chi2_; }
	double GetSumEt() { return SumEt_; }
	int GetIdHadb() { return IdHadb_;}
	int GetIdWp() { return IdWp_; }
	int GetIdWq() { return IdWq_; }
	int GetIdLepb() { return IdLepb_;}
	void SetIdHadb(int id) { IdHadb_ = id;}
	void SetIdWp(int id) { IdWp_ = id; }
	void SetIdWq(int id) { IdWq_ = id; }
	void SetIdLepb(int id) { IdLepb_ = id;}
	
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

	double chi2_;
	double SumEt_;
	double minMassLepW_;
	double maxMassLepW_;
	double minMassHadW_;
	double maxMassHadW_;
	
	double minMassLepTop_;
	double maxMassLepTop_;

	double MW;
	double Mtop;

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

	std::map< int, std::string > Combinatorics(int k, int max = 6);
	std::map< int, std::string > NestedCombinatorics();

	void FourJetsCombinations(std::vector<TLorentzVector> jets);
	void SetMaxNJets(int n) { maxNJets_ = n; }
	Combo GetCombination(int n=0);
	Combo GetCombinationSumEt(int n=0);
	int GetNumberOfCombos() { return ( (int)allCombos_.size() ); } 
	//void SetCandidate( std::vector< TLorentzVector > JetCandidates );
	
	void SetLeptonicW( TLorentzVector LepW ) { theLepW_ = LepW; }

	void SetMinMassLepW( double mass ) { minMassLepW_ = mass; }
	void SetMaxMassLepW( double mass ) { maxMassLepW_ = mass; }
	void SetMinMassHadW( double mass ) { minMassHadW_ = mass; }
	void SetMaxMassHadW( double mass ) { maxMassHadW_ = mass; }
	void SetMinMassLepTop( double mass ) { minMassLepTop_ = mass; }
	void SetMaxMassLepTop( double mass ) { maxMassLepTop_ = mass; }

	void Clear();

	std::vector< TLorentzVector > TwoCombos();
	std::vector< TLorentzVector > ThreeCombos();

	void RemoveDuplicates( bool option) { removeDuplicates_ = option; }

	std::vector< TLorentzVector > GetComposites();
	void AnalyzeCombos();


  private:

	//int kcombos_;
	//int maxcombos_;
	std::map< int, std::string > Template4jCombos_;
	std::map< int, std::string > Template5jCombos_;
	std::map< int, std::string > Template6jCombos_;
	std::map< int, std::string > Template7jCombos_;

	int maxNJets_;
	
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
