#ifndef RecoB_HistoCompare_h
#define RecoB_HistoCompare_h

/**_________________________________________________________________
   class:   HistoCompare.h
   package: Validation/RecoB
   

 author: Victor Bazterra, UIC
         Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: HistoCompare.h,v 1.2 2007/02/14 20:09:41 yumiceva Exp $

________________________________________________________________**/


#include "TFile.h"
#include "TH1.h"
#include "TString.h"

#include <iostream>
#include <string>
#include <vector>

class HistoCompare {

  public:
	HistoCompare();
	HistoCompare(TString refFilename);

	~HistoCompare();

	TH1* Compare(TH1* h, TString hname);

	void SetReferenceFilename(TString filename) {
	  refFilename_ = filename;
	  //if (refFile_) delete refFile_;
	  refFile_ = new TFile(refFilename_);
	  if (refFile_->IsZombie()) {
		  std::cout << " Error openning file " << refFilename_ << std::endl;
		  std::cout << " we will not compare histograms. " << std::endl;
		  do_nothing_ = true;
	  }
	  std::cout << " open file" << std::endl;
	};

	void SetChi2Test(bool test=true) {
		setChi2Test_ = test;
	};
	
	void SetKGTest(bool test=true) {
		setKGTest_ = test;
	};
	
	double GetResult() { return result_; };

  private:

	bool setChi2Test_;
	bool setKGTest_;
	double result_;
	bool do_nothing_;
	
	TH1 *resHisto_;
	TH1 *refHisto_;
	TFile *refFile_;
	
	TString refFilename_;

	//std::map<std::string, TH1*> histomap_;
	
};

#endif
