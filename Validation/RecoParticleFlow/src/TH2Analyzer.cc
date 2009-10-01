#include "Validation/RecoParticleFlow/interface/TH2Analyzer.h"

#include <TH2D.h>
#include <TH1D.h>

#include <string>
#include <iostream>

using namespace std; 

void TH2Analyzer::Eval( int rebinFactor ) {
  
  Reset();

  string bname = hist2D_->GetName();

  string rebinName = bname + "_rebin";
  rebinnedHist2D_ = (TH2D*) hist2D_ -> Clone( rebinName.c_str() );
  rebinnedHist2D_->RebinX( rebinFactor );

  string averageName = bname + "_average";  
  average_ = new TH1D( averageName.c_str(),"arithmetic average", 
		       rebinnedHist2D_->GetNbinsX(),
		       rebinnedHist2D_->GetXaxis()->GetXmin(),
		       rebinnedHist2D_->GetXaxis()->GetXmax() );
  
  string rmsName = bname + "_RMS";  
  RMS_ = new TH1D( rmsName.c_str(), "RMS",
		   rebinnedHist2D_->GetNbinsX(),
		   rebinnedHist2D_->GetXaxis()->GetXmin(),
		   rebinnedHist2D_->GetXaxis()->GetXmax() );



  ProcessSlices( rebinnedHist2D_ );
}

void TH2Analyzer::Reset() {
  if( rebinnedHist2D_ ) delete rebinnedHist2D_;
  if( average_ ) delete average_;
  if( RMS_ ) delete RMS_;
  
  for( unsigned i=0; i<parameters_.size(); ++i) {
    delete parameters_[i];
  }
  
  parameters_.clear();
}


void TH2Analyzer::ProcessSlices( const TH2D* histo) {
  
  TH1::AddDirectory(0);

  for( int i=1; i<=histo->GetNbinsX(); ++i) {
    TH1D* proj =  histo->ProjectionY("toto", i, i);
    const double mean = proj->GetMean();
    const double rms = proj->GetRMS();
    // cout<<mean<<" "<<rms<<endl;
    average_->SetBinContent( i, mean);
    average_->SetBinError( i, proj->GetMeanError());
    RMS_->SetBinContent(i, rms);
    RMS_->SetBinError(i, proj->GetRMSError());
    ProcessSlice( proj );
  }

  TH1::AddDirectory(1);
} 


void TH2Analyzer::ProcessSlice( const TH1D* histo ) const {
  
}
