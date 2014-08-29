#include "Validation/RecoParticleFlow/interface/TH2Analyzer.h"

#include "TH2D.h"
#include "TH1D.h"
#include "TF1.h"
#include "TPad.h"

#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>

using namespace std; 

// remove? 
void TH2Analyzer::Eval( int rebinFactor ) {
  //std::cout << "Eval!" << std::endl;
  
  Reset();

  const string bname = hist2D_->GetName();
  const string rebinName = bname + "_rebin";
  rebinnedHist2D_ = (TH2D*) hist2D_ -> Clone( rebinName.c_str() );
  rebinnedHist2D_->RebinX( rebinFactor );

  const string averageName = bname + "_average";  
  average_ = new TH1D( averageName.c_str(),"arithmetic average", 
		       rebinnedHist2D_->GetNbinsX(),
		       rebinnedHist2D_->GetXaxis()->GetXmin(),
		       rebinnedHist2D_->GetXaxis()->GetXmax() );
  
  const string rmsName = bname + "_RMS";  
  RMS_ = new TH1D( rmsName.c_str(), "RMS",
		   rebinnedHist2D_->GetNbinsX(),
		   rebinnedHist2D_->GetXaxis()->GetXmin(),
		   rebinnedHist2D_->GetXaxis()->GetXmax() );

  const string sigmaGaussName = bname + "_sigmaGauss"; 
  sigmaGauss_ = new TH1D(sigmaGaussName.c_str(), "sigmaGauss",
			 rebinnedHist2D_->GetNbinsX(),
			 rebinnedHist2D_->GetXaxis()->GetXmin(),
			 rebinnedHist2D_->GetXaxis()->GetXmax() );

  const string meanXName = bname + "_meanX"; 
  meanXslice_ = new TH1D(meanXName.c_str(), "meanX",
			 rebinnedHist2D_->GetNbinsX(),
			 rebinnedHist2D_->GetXaxis()->GetXmin(),
			 rebinnedHist2D_->GetXaxis()->GetXmax() );

  ProcessSlices( rebinnedHist2D_ );
}

void TH2Analyzer::Reset() {
  if ( rebinnedHist2D_ ) delete rebinnedHist2D_;
  if ( average_ ) delete average_;
  if ( RMS_ ) delete RMS_;
  if ( sigmaGauss_ ) delete sigmaGauss_;
  if ( meanXslice_ ) delete meanXslice_;

  //for( unsigned i=0; i<parameters_.size(); ++i) {
  //  delete parameters_[i];
  //}
  
  //parameters_.clear();
}

void TH2Analyzer::Eval(const int rebinFactor, const int binxmin,
		       const int binxmax, const bool cst_binning)
{
  Reset();
  const string bname = hist2D_->GetName();
  const string rebinName = bname + "_rebin";

  if (binxmax>hist2D_->GetNbinsX())
    {
      std::cout << "Error: TH2Analyzer.cc: binxmax>hist2D_->GetNbinsX()" << std::endl;
      return;
    }

  if (cst_binning)
    {
      //std::cout << "hist2D_->GetXaxis()->GetBinLowEdge(" << binxmin << ") = " << hist2D_->GetXaxis()->GetBinLowEdge(binxmin) << std::endl;
      //std::cout << "hist2D_->GetXaxis()->GetBinUpEdge(" << binxmax << ") = " << hist2D_->GetXaxis()->GetBinUpEdge(binxmax) << std::endl;
      //std::cout << "hist2D_->GetNbinsY() = " << hist2D_->GetNbinsY() << std::endl;
      //std::cout << "hist2D_->GetYaxis()->GetXmin() = " << hist2D_->GetYaxis()->GetXmin() << std::endl;
      //std::cout << "hist2D_->GetYaxis()->GetXmax() = " << hist2D_->GetYaxis()->GetXmax() << std::endl;
      rebinnedHist2D_ = new TH2D(rebinName.c_str(),"rebinned histo",
				 binxmax-binxmin+1,
				 hist2D_->GetXaxis()->GetBinLowEdge(binxmin),
				 hist2D_->GetXaxis()->GetBinUpEdge(binxmax),
				 hist2D_->GetNbinsY(),
				 hist2D_->GetYaxis()->GetXmin(),
				 hist2D_->GetYaxis()->GetXmax() );
      for (int binyc=1;binyc<hist2D_->GetNbinsY()+1;++binyc)
	{
	  for (int binxc=binxmin;binxc<binxmax+1;++binxc)
	    {
	      //std::cout << "hist2D_->GetBinContent(" << binxc << "," << binyc << ") = " << hist2D_->GetBinContent(binxc,binyc) << std::endl;
	      //std::cout << "hist2D_->GetBinError(" << binxc << "," << binyc << ") = " << hist2D_->GetBinError(binxc,binyc) << std::endl;
	      //std::cout << "binxc-binxmin+1 = " << binxc-binxmin+1 << std::endl;
	      rebinnedHist2D_->SetBinContent(binxc-binxmin+1,binyc,hist2D_->GetBinContent(binxc,binyc));
	      rebinnedHist2D_->SetBinError(binxc-binxmin+1,binyc,hist2D_->GetBinError(binxc,binyc));
	    }
	}
      rebinnedHist2D_->RebinX( rebinFactor );
  
      const string averageName = bname + "_average";  
      average_ = new TH1D( averageName.c_str(),"arithmetic average", 
			   rebinnedHist2D_->GetNbinsX(),
			   rebinnedHist2D_->GetXaxis()->GetXmin(),
			   rebinnedHist2D_->GetXaxis()->GetXmax() );
    
      const string rmsName = bname + "_RMS";  
      RMS_ = new TH1D( rmsName.c_str(), "RMS",
		       rebinnedHist2D_->GetNbinsX(),
		       rebinnedHist2D_->GetXaxis()->GetXmin(),
		       rebinnedHist2D_->GetXaxis()->GetXmax() );
  
      const string sigmaGaussName = bname + "_sigmaGauss"; 
      sigmaGauss_ = new TH1D(sigmaGaussName.c_str(), "sigmaGauss",
			     rebinnedHist2D_->GetNbinsX(),
			     rebinnedHist2D_->GetXaxis()->GetXmin(),
			     rebinnedHist2D_->GetXaxis()->GetXmax() );
  
      const string meanXName = bname + "_meanX"; 
      meanXslice_ = new TH1D(meanXName.c_str(), "meanX",
			     rebinnedHist2D_->GetNbinsX(),
			     rebinnedHist2D_->GetXaxis()->GetXmin(),
			     rebinnedHist2D_->GetXaxis()->GetXmax() );
    }
  else
    {
      // binning is not constant, but made to obtain almost the same number of events in each bin

      //std::cout << "binxmax-binxmin+1 = " << binxmax-binxmin+1 << std::endl;
      //std::cout << "rebinFactor = " << rebinFactor << std::endl;
      //std::cout << "(binxmax-binxmin+1)/rebinFactor = " << (binxmax-binxmin+1)/rebinFactor << std::endl;
      //std::cout << "((binxmax-binxmin+1)%rebinFactor) = " << ((binxmax-binxmin+1)%rebinFactor) << std::endl;
      //std::cout << "abs((binxmax-binxmin+1)/rebinFactor) = " << std::abs((binxmax-binxmin+1)/rebinFactor) << std::endl;

      unsigned int nbin=0;
      if (((binxmax-binxmin+1)%rebinFactor)!=0.0)
	{
	  nbin=std::abs((binxmax-binxmin+1)/rebinFactor)+1;
	}
      else nbin=(binxmax-binxmin+1)/rebinFactor;

      double *xlow = new double[nbin+1];
      int *binlow = new int[nbin+1];

      TH1D* h0_slice1 = hist2D_->ProjectionY("h0_slice1", binxmin, binxmax, "");
      const unsigned int totalNumberOfEvents=static_cast<unsigned int>(h0_slice1->GetEntries());
      //std::cout << "totalNumberOfEvents = " << totalNumberOfEvents << std::endl;
      delete h0_slice1;

      unsigned int neventsc=0;
      unsigned int binXmaxc=binxmin+1;
      xlow[0]=hist2D_->GetXaxis()->GetBinLowEdge(binxmin);
      binlow[0]=binxmin;
      for (unsigned int binc=1;binc<nbin;++binc)
	{
	  while (neventsc<binc*totalNumberOfEvents/nbin)
	    {
	      TH1D* h0_slice1c = hist2D_->ProjectionY("h0_slice1",binxmin, binXmaxc, "");
	      neventsc=static_cast<unsigned int>(h0_slice1c->GetEntries());
	      //        //std::cout << "FL : neventsc = " << neventsc << std::endl;
	      //        //std::cout << "FL : binXmaxc = " << binXmaxc << std::endl;
	      ++binXmaxc;
	      delete h0_slice1c;
	    }
	  //std::cout << "binXmaxc-1 = " << binXmaxc-1 << std::endl;
	  binlow[binc]=binXmaxc-1;
	  xlow[binc]=hist2D_->GetXaxis()->GetBinLowEdge(binXmaxc-1);
	}
      xlow[nbin]=hist2D_->GetXaxis()->GetBinUpEdge(binxmax);
      binlow[nbin]=binxmax;

      //for (unsigned int binc=0;binc<nbin+1;++binc)
      //{
      //  std::cout << "xlow[" << binc << "] = " << xlow[binc] << std::endl;
      //}

      rebinnedHist2D_ = new TH2D(rebinName.c_str(),"rebinned histo",
				 nbin, xlow, hist2D_->GetNbinsY(),
				 hist2D_->GetYaxis()->GetXmin(),
				 hist2D_->GetYaxis()->GetXmax() );
      for (int binyc=1;binyc<hist2D_->GetNbinsY()+1;++binyc)
	{
	  for (unsigned int binxc=1;binxc<nbin+1;++binxc)
	    {
	      double sum=0.0;
	      for (int binxh2c=binlow[binxc-1];binxh2c<binlow[binxc];++binxh2c)
		{
		  sum+=hist2D_->GetBinContent(binxh2c,binyc);
		}
	      rebinnedHist2D_->SetBinContent(binxc,binyc,sum);
	    }
	}
  
      const string averageName = bname + "_average";  
      average_ = new TH1D( averageName.c_str(),"arithmetic average", nbin, xlow);
    
      const string rmsName = bname + "_RMS";  
      RMS_ = new TH1D( rmsName.c_str(), "RMS", nbin, xlow);
  
      const string sigmaGaussName = bname + "_sigmaGauss"; 
      sigmaGauss_ = new TH1D(sigmaGaussName.c_str(), "sigmaGauss", nbin, xlow);
  
      const string meanXName = bname + "_meanX"; 
      meanXslice_ = new TH1D(meanXName.c_str(), "meanX", nbin, xlow);
      delete [] xlow;
      delete [] binlow;
    }
  ProcessSlices( rebinnedHist2D_ );
}

void TH2Analyzer::ProcessSlices( const TH2D* histo) {
  
  //std::cout << "ProcessSlices!" << std::endl;

  TH1::AddDirectory(0);

  for( int i=1; i<=histo->GetNbinsX(); ++i) {
    TH1D* proj =  histo->ProjectionY("toto", i, i);
    const double mean = proj->GetMean();
    const double rms = proj->GetRMS();
    //std::cout << "mean = " << mean << std::endl;
    //std::cout << "rms = " << rms << std::endl;
    average_->SetBinContent( i, mean);
    average_->SetBinError( i, proj->GetMeanError());
    RMS_->SetBinContent(i, rms);
    RMS_->SetBinError(i, proj->GetRMSError());

    const double error=(histo->GetXaxis()->GetBinUpEdge(i)-histo->GetXaxis()->GetBinLowEdge(i))/2.0;
    //std::cout << "error = " << error << std::endl;
    meanXslice_->SetBinContent(i, histo->GetXaxis()->GetBinLowEdge(i)+error);
    meanXslice_->SetBinError(i, error);
    //std::cout << "histo->GetXaxis()->GetBinLowEdge(" << i << ") = "
    //          << histo->GetXaxis()->GetBinLowEdge(i) << std::endl;
    //std::cout << "meanXslice_->GetBinError(" << i << ") = "
    //	      << meanXslice_->GetBinError(i) << std::endl;
    ProcessSlice(i, proj );
    delete proj;
  }

  TH1::AddDirectory(1);
} 


void TH2Analyzer::ProcessSlice(const int i, TH1D* proj ) const {

  //const double mean = proj->GetMean();
  const double rms = proj->GetRMS();
  //std::cout << "FL: mean = " << mean << std::endl;
  //std::cout << "FL: rms = " << rms << std::endl;

  if (rms!=0.0)
    {
      const double fitmin=proj->GetMean()-proj->GetRMS();
      const double fitmax=proj->GetMean()+proj->GetRMS();
  
      //std::cout << "i = " << i << std::endl;
      //std::cout << "fitmin = " << fitmin << std::endl;
      //std::cout << "fitmax = " << fitmax << std::endl;
  
      //proj->Draw();
      TF1* f1= new TF1("f1", "gaus", fitmin, fitmax);
      f1->SetParameters(proj->GetRMS(),proj->GetMean(),proj->GetBinContent(proj->GetMaximumBin()));
      proj->Fit(f1,"R", "", proj->GetXaxis()->GetXmin(), proj->GetXaxis()->GetXmax());
  
      //std::ostringstream oss;
      //oss << i;
      //const std::string plotfitname="Plots/fitbin_"+oss.str()+".eps";
      //gPad->SaveAs( plotfitname.c_str() );
      //std::cout << "param1 = " << f1->GetParameter(1) << std::endl;
      //std::cout << "param2 = " << f1->GetParameter(2) << std::endl;
      //std::cout << "paramError2 = " << f1->GetParError(2) << std::endl;
  
      sigmaGauss_->SetBinContent(i, f1->GetParameter(2));
      sigmaGauss_->SetBinError(i, f1->GetParError(2));
      delete f1;
    }
  else
    {
      sigmaGauss_->SetBinContent(i, 0.0);
      sigmaGauss_->SetBinError(i, 0.0);
    }
}
