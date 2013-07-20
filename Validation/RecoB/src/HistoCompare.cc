/**_________________________________________________________________
   class:   HistoCompare.cc
   package: Validation/RecoB
   

 author: Victor Bazterra, UIC
         Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: HistoCompare.cc,v 1.3 2007/02/14 20:20:08 yumiceva Exp $

________________________________________________________________**/

#include "Validation/RecoB/interface/HistoCompare.h"

#include <iostream>

HistoCompare::HistoCompare() {
  result_ = 0;
  do_nothing_ = false;
}

HistoCompare::HistoCompare(TString refFilename) {

	result_ = 0;
	do_nothing_ = false;
	// open reference file
	refFile_ = new TFile(refFilename);
	if (refFile_->IsZombie()) {
		std::cout << " Error openning file " << refFilename << std::endl;
		std::cout << " we will not compare histograms. " << std::endl;
		do_nothing_ = true;
	} else {
		std::cout << " open file " << refFilename << std::endl;
	}
	
	setChi2Test_ = true;
	
}

HistoCompare::~HistoCompare() {
}


TH1* HistoCompare::Compare(TH1 *h, TString hname) {

	if (do_nothing_) return 0;
	
	//std::cout << "histo: " << h->GetName() << " entries: "<< h->GetEntries() << std::endl;
	//std::cout << "hname: " << hname << std::endl;

	
	//create a residual histogram
	//if (resHisto_) delete resHisto_;
	resHisto_ = (TH1*) h->Clone(TString(h->GetName())+"_residuals");
	resHisto_->Reset();

	// clone input histogram
	TH1 *htemp = (TH1*) h->Clone(TString(h->GetName()));

	refFile_->cd();
	//std::cout << "get reference" << std::endl;
	refHisto_ = (TH1*) gDirectory->Get( hname );
	//std::cout << "historef: " << refHisto_->GetName() << " entries: "<< refHisto_->GetEntries() << std::endl;
	//std::cout << "name: " << refHisto_->GetName() << std::endl;


	// normalize histograms
	htemp->Scale(1./htemp->Integral());
	refHisto_->Scale(1./refHisto_->Integral());
	
	if (setChi2Test_) {
		result_ = refHisto_->Chi2Test(htemp,"UFOF");
	}
	if (setKGTest_) {
		result_ = refHisto_->KolmogorovTest(htemp,"UO");
	}

	
	resHisto_->Add( htemp, refHisto_, 1., -1.);
	
	//std::cout << " residual done." << std::endl;
	
	return resHisto_;
	
}
