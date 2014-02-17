/**_________________________________________________________________
   class:   JetCombinatorics.cc
   package: Analyzer/TopTools


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: JetCombinatorics.cc,v 1.4 2012/10/11 14:25:45 eulisse Exp $

________________________________________________________________**/


#include "JetCombinatorics.h"
#include "TMath.h"

#include <iostream>

std::string itoa(int i) {
  char temp[20];
  sprintf(temp,"%d",i);
  return((std::string)temp);
}



//_______________________________________________________________
JetCombinatorics::JetCombinatorics() {

	this->Clear();
		
	minMassLepW_ = -999999.;
	maxMassLepW_ = 999999.;
	minMassHadW_ = -999999.;
	maxMassHadW_ = 999999.;
	minMassLepTop_ = -999999.;
	maxMassLepTop_ = 999999.;
	
	minPhi_ = -1.;
	removeDuplicates_ = true;
	maxNJets_ = 9999;
	verbosef = false;
	UsebTagging_ = false;
	UseMtop_ = true;
	SigmasTypef = 0;
	UseFlv_ = false;
	
	Template4jCombos_ = NestedCombinatorics(); // 12 combinations
	Template5jCombos_ = Combinatorics(4,5); // 5 combinations of 4 combos
	Template6jCombos_ = Combinatorics(4,6); // 15 combinations of 4 combos
	Template7jCombos_ = Combinatorics(4,7); // xx combinations of 4 combos
	
}

//_______________________________________________________________
JetCombinatorics::~JetCombinatorics() {
  this->Clear();
}

//_______________________________________________________________
void JetCombinatorics::Clear() {

	allCombos_.clear();
	allCombosSumEt_.clear();
	Template4jCombos_.clear();
	Template5jCombos_.clear();
	Template6jCombos_.clear();
	Template7jCombos_.clear();
	cand1_.clear();
	
}
	

//_______________________________________________________________
std::map< int, std::string > JetCombinatorics::Combinatorics(int n, int max) {

	// find a combinatorics template
	// This is a simple stupid function to make algebratic combinatorics

	int kcombos = n;
	int maxcombos = max;

	std::string list;

	for ( int m=0; m<maxcombos; m++) { list = list + (itoa(m));}

	std::string seed;
	for ( int m=0; m<kcombos; m++) { seed = seed + (itoa(m));}

	
	std::map< int, std::string > aTemplateCombos;
	aTemplateCombos.clear();
	
	aTemplateCombos[0] = seed;

	int i = 0;
	int totalmatches = seed.size();
	int totalIte = list.size();

	for ( int ite = 0; ite < ((int)totalIte); ite++) {

		//cout << "iteration " << ite << endl;
		//i = 0;
		//for ( Itevec = seed.begin(); Itevec != seed.end(); ++Itevec) {
		for ( i=0; i< (int) totalmatches; i++) {

			std::string newseed = aTemplateCombos[ite];
			std::string newseed2;
			/*
			cout << "newseed size= " << newseed.size() << " : ";
			for (std::vector< std::string>::iterator iite = newseed.begin();
				 iite != newseed.end(); ++iite) {

				cout << *iite << " ";
			}
			cout << endl;
			*/
			for ( int itemp=0; itemp<(int)newseed.size(); itemp++) {
				if (itemp!=i) newseed2 = newseed2 + (newseed[itemp]);
			}
			/*
			cout << "newseed2: ";
			for (std::vector< std::string>::iterator iite = newseed2.begin();
				 iite != newseed2.end(); ++iite) {

				cout << *iite << " ";
			}
			cout << endl;
			*/
			for ( int j=0; j<(int) list.size(); j++) {
				//cout << " j = " << j << endl;
				bool Isnewelement = true;
				std::string newelement = "0";
				//bool Isnewcombo = true;
				for (int k=0; k< (int)newseed2.size(); k++) {
					if ( list[j] == newseed2[k] ) Isnewelement = false;
				}
				if (Isnewelement) {

					newelement = list[j];
					//cout << "new element: " << newelement << endl;

					std::string candseed = newseed2;
					candseed = candseed + newelement;

					bool IsnewCombo = true;
					for (int ic=0; ic<(int)aTemplateCombos.size(); ++ic ) {

						int nmatch = 0;
						for ( int ij=0; ij<(int)(aTemplateCombos[ic]).size(); ij++) {

							for (int ik=0; ik<(int)candseed.size(); ik++) {
								if ( candseed[ik] == aTemplateCombos[ic][ij] ) nmatch++;
							}
						}
						if (nmatch == (int)totalmatches)
							IsnewCombo = false;

					}
					if (IsnewCombo) {
						//cout << "new combo"<< " before combo size=" << aTemplateCombos.size() << endl;
						aTemplateCombos[(int)aTemplateCombos.size()] = candseed;
						//cout << " after size = " << aTemplateCombos.size() << endl;
					}
				}

			}
		}
	}//close iterations

	// debug info
	
	//std::cout << " build templates for total combos = " << aTemplateCombos.size() << std::endl;
	//std::cout << " template combos: " << std::endl;
	//for (size_t ic=0; ic != aTemplateCombos.size(); ++ic) {

	//std::cout << aTemplateCombos[ic] << std::endl;
	//}
	
	return aTemplateCombos;
	
	
	
}


//______________________________________________________________
std::map< int, std::string > JetCombinatorics::NestedCombinatorics() {

	// build by hand 12 combinations for semileptonic top decays

	std::map< int, std::string > aTemplateCombos;
	aTemplateCombos.clear();
	
	aTemplateCombos[0] = "0123";
	aTemplateCombos[1] = "0132";
	aTemplateCombos[2] = "0213";
	aTemplateCombos[3] = "0231";
	aTemplateCombos[4] = "0312";
	aTemplateCombos[5] = "0321";
	aTemplateCombos[6] = "1203";
	aTemplateCombos[7] = "1230";
	aTemplateCombos[8] = "1302";
	aTemplateCombos[9] = "1320";
	aTemplateCombos[10] = "2301";
	aTemplateCombos[11] = "2310";
		
	return aTemplateCombos;

}

//______________________________________________________________
void JetCombinatorics::FourJetsCombinations(std::vector<TLorentzVector> jets, std::vector<double> bdiscriminators ) {


	int n = 0; // total number of combos
	std::map< Combo, int, minChi2 > allCombos;
	std::map< Combo, int, maxSumEt > allCombosSumEt;
	
	std::map< int, std::string > aTemplateCombos;
	aTemplateCombos.clear();

	if ( jets.size() == 4 ) aTemplateCombos[0] = std::string("0123");
	if ( jets.size() == 5 ) aTemplateCombos = Template5jCombos_;
	if ( jets.size() == 6 ) aTemplateCombos = Template6jCombos_;
	if ( jets.size() == 7 ) aTemplateCombos = Template7jCombos_;	

	// force to use only 4 jets
	if ( maxNJets_ == 4 ) aTemplateCombos[0] = std::string("0123");
	
	if (verbosef) std::cout << "[JetCombinatorics] size of vector of jets = " << jets.size() << std::endl;
	
	for (size_t ic=0; ic != aTemplateCombos.size(); ++ic) {

		if (verbosef) std::cout << "[JetCombinatorics] get 4 jets from the list, cluster # " << ic << "/"<< aTemplateCombos.size()-1 << std::endl;
		
		// get a template
		std::string aTemplate = aTemplateCombos[ic];

		if (verbosef) std::cout << "[JetCombinatorics] template of 4 jets = " << aTemplate << std::endl;
		
		// make a list of 4 jets
		std::vector< TLorentzVector > the4jets;
		std::vector< int > the4Ids;
		std::vector< double > thebdisc;
		std::vector< double > theFlvCorr;
		//the4jets[0] = jets[0];
		
		for (int ij=0; ij<4; ij++) {
			//std::cout << "ij= " << ij << std::endl;
			//std::cout << "atoi = " << atoi((aTemplate.substr(0,1)).c_str()) << std::endl;
			//std::cout << "jets[].Pt = " << jets[ij].Pt() << std::endl;
			int tmpi = atoi((aTemplate.substr(ij,1)).c_str());
			//std::cout << "tmpi= " << tmpi << std::endl;
			the4jets.push_back(jets[tmpi]);
			the4Ids.push_back(tmpi);
			if ( UsebTagging_ ) thebdisc.push_back( bdiscriminators[tmpi] );
			if ( UseFlv_ ) theFlvCorr.push_back( flavorCorrections_[tmpi] );
		}

		if (verbosef) std::cout<< "[JetCombinatorics] with these 4 jets, make 12 combinations: " <<std::endl;

		//std::cout << " the4jets[ij].size = " << the4jets.size() << std::endl;
			
		for (size_t itemplate=0; itemplate!= Template4jCombos_.size(); ++itemplate) {
			
			std::string a4template = Template4jCombos_[itemplate];

			if (verbosef) std::cout << "[JetCombinatorics] ==> combination: " << a4template << " is # " << itemplate << "/"<<  Template4jCombos_.size()-1 << std::endl;
			
			Combo acombo;
			
			acombo.SetWp( the4jets[atoi((a4template.substr(0,1)).c_str())] );
			acombo.SetWq( the4jets[atoi((a4template.substr(1,1)).c_str())] );
			acombo.SetHadb( the4jets[atoi((a4template.substr(2,1)).c_str())] );
			acombo.SetLepb( the4jets[atoi((a4template.substr(3,1)).c_str())] );
			acombo.SetLepW( theLepW_ );

			acombo.SetIdWp( the4Ids[atoi((a4template.substr(0,1)).c_str())] );
			acombo.SetIdWq( the4Ids[atoi((a4template.substr(1,1)).c_str())] );
			acombo.SetIdHadb( the4Ids[atoi((a4template.substr(2,1)).c_str())] );
			acombo.SetIdLepb( the4Ids[atoi((a4template.substr(3,1)).c_str())] );
			//std::cout << " acombo setup" << std::endl;

			if ( UseFlv_ ) {
				acombo.SetFlvCorrWp( theFlvCorr[atoi((a4template.substr(0,1)).c_str())] );
				acombo.SetFlvCorrWq( theFlvCorr[atoi((a4template.substr(1,1)).c_str())] );
				acombo.SetFlvCorrHadb( theFlvCorr[atoi((a4template.substr(2,1)).c_str())] );
				acombo.SetFlvCorrLepb( theFlvCorr[atoi((a4template.substr(3,1)).c_str())] );
				acombo.ApplyFlavorCorrections();
			}
			if ( UsebTagging_ ) {

				acombo.Usebtagging();
				acombo.SetbDiscPdf(bTagPdffilename_);
				acombo.SetWp_disc( thebdisc[atoi((a4template.substr(0,1)).c_str())] );
				acombo.SetWq_disc( thebdisc[atoi((a4template.substr(1,1)).c_str())] );
				acombo.SetHadb_disc( thebdisc[atoi((a4template.substr(2,1)).c_str())] );
				acombo.SetLepb_disc( thebdisc[atoi((a4template.substr(3,1)).c_str())] );
				
			}

			acombo.UseMtopConstraint(UseMtop_);
			// choose value of sigmas
			acombo.SetSigmas(SigmasTypef);

			acombo.analyze();

			if (verbosef) {

			  std::cout << "[JetCombinatorics] ==> combination done:" << std::endl;
			  acombo.Print();
			}

			// invariant mass cuts
			TLorentzVector aHadWP4 = acombo.GetHadW();
			TLorentzVector aLepWP4 = acombo.GetLepW();
			TLorentzVector aLepTopP4=acombo.GetLepTop();
			
			if ( ( aHadWP4.M() > minMassHadW_ && aHadWP4.M() < maxMassHadW_ ) &&
				 ( aLepWP4.M() > minMassLepW_ && aLepWP4.M() < maxMassLepW_ ) &&
				 ( aLepTopP4.M() > minMassLepTop_ && aLepTopP4.M() < maxMassLepTop_) ) {
			
				allCombos[acombo] = n;
				allCombosSumEt[acombo] = n;
			
				n++;
			}
		
		}
	}

	allCombos_ = allCombos;
	allCombosSumEt_ = allCombosSumEt;
       
}

Combo JetCombinatorics::GetCombination(int n) {

	int j = 0;
	Combo a;
	for ( std::map<Combo,int,minChi2>::const_iterator ite=allCombos_.begin();
		  ite!=allCombos_.end(); ++ite) {
		
		if (j == n) a = ite->first;
		j++;
	}

	return a;

	
}

Combo JetCombinatorics::GetCombinationSumEt(int n) {

	int j = 0;
	Combo a;
	for ( std::map<Combo,int,maxSumEt>::const_iterator ite=allCombosSumEt_.begin();
		  ite!=allCombosSumEt_.end(); ++ite) {
		
		if (j == n) a = ite->first;
		j++;
	}

	return a;

	
}
