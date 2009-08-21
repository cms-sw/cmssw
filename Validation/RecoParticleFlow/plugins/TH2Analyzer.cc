#include "Validation/RecoParticleFlow/plugins/TH2Analyzer.h"
#include <iostream>
#include <sstream>
#include "TPad.h"

ClassImp(TH2Analyzer)

TH2Analyzer::TH2Analyzer()
{

}

TH2Analyzer::~TH2Analyzer()
{

}

void TH2Analyzer::checkBinning() const
{
  //std::cout << "GetNbinsX() = " << GetNbinsX() << std::endl;
  //std::cout << "GetXmax = " << GetXaxis()->GetXmax() << std::endl;
  if (GetNbinsX()!=GetXaxis()->GetXmax())
    std::cout << "Warning in TH2Analyzer: nbin != xmax in input 2D histo" << std::endl;
}


void TH2Analyzer::XSlice(const std::string Xvar, TH1D* gr, const unsigned int binxmin,
			 const unsigned int binxmax, const unsigned int nbin,
			 const std::string binning_option, const std::string gauss,
			 const int range, const double fitmin, const double fitmax,
			 const std::string fit_plot_name) const
{
  checkBinning();
  double *xlow = new double[nbin+1];
  binning_computation(xlow,nbin,binxmin,binxmax,binning_option);
  gr->SetBins(nbin, xlow);
  for (unsigned int binc=0;binc<nbin;++binc)
  {
    const int binxminc=static_cast<int>(xlow[binc]);
    const int binxmaxc=static_cast<int>(xlow[binc+1]);
    TH1D* h0_slice = this->ProjectionY("h0_slice",binxminc, binxmaxc, "");
    //std::cout << "FL: gauss = " << gauss << std::endl;
    if (gauss=="Gauss")
    {
      // fit
      h0_slice->Rebin(range);
      h0_slice->Draw();
      TF1* f1= new TF1("f1","gaus",fitmin,fitmax);
      f1->SetParameters(h0_slice->GetRMS(1),h0_slice->GetMean(1),h0_slice->GetBinContent(h0_slice->GetMaximumBin()));
      h0_slice->Fit("f1","R");
      std::ostringstream oss;
      oss << binc;
      const std::string plotfitname="Plots/fitbin_"+fit_plot_name+"_"+oss.str()+".eps";
      gPad->SaveAs( plotfitname.c_str() );
      //std::cout << "param1 = " << f1->GetParameter(1) << std::endl;
      if (Xvar=="Mean")
      {
	gr->SetBinContent(binc+1,f1->GetParameter(1));
	gr->SetBinError(binc+1,f1->GetParError(1));
      }
      else if (Xvar=="Sigma")
      {
	gr->SetBinContent(binc+1,f1->GetParameter(2));
	gr->SetBinError(binc+1,f1->GetParError(2));
      }
      else
      {
	std::cout << "Error : TH2Analyzer: Xvar unknown : " << Xvar << std::endl;
      }
    }
    else
    {
      if (Xvar=="Mean")
      {
	gr->SetBinContent(binc+1,h0_slice->GetMean(1));
	gr->SetBinError(binc+1,h0_slice->GetMeanError(1));
      }
      else if (Xvar=="Sigma")
      {
	gr->SetBinContent(binc+1,h0_slice->GetRMS(1));
	gr->SetBinError(binc+1,h0_slice->GetRMSError(1));
      }
      else
      {
	std::cout << "Error : TH2Analyzer: Xvar unknown : " << Xvar << std::endl;
      }
    }
    delete h0_slice;
  }
  delete [] xlow;
}

void TH2Analyzer::MeanSlice(TH1D* gr, const unsigned int binxmin, const unsigned int binxmax,
			    const unsigned int nbin, const std::string binning_option) const
{
  XSlice("Mean", gr, binxmin, binxmax, nbin, binning_option, "", 0, 0.0, 0.0, "");
}

void TH2Analyzer::SigmaSlice(TH1D* gr, const unsigned int binxmin, const unsigned int binxmax,
			    const unsigned int nbin, const std::string binning_option) const
{
  XSlice("Sigma", gr, binxmin, binxmax, nbin, binning_option, "", 0, 0.0, 0.0, "");
}

void TH2Analyzer::MeanGaussSlice(TH1D* gr, const unsigned int binxmin, const unsigned int binxmax,
				 const unsigned int nbin, const std::string binning_option,
				 const int range, const double fitmin, const double fitmax,
				 const std::string fit_plot_name) const
{
  XSlice("Mean", gr, binxmin, binxmax, nbin, binning_option, "Gauss", range, fitmin, fitmax, fit_plot_name);
}

void TH2Analyzer::SigmaGaussSlice(TH1D* gr, const unsigned int binxmin, const unsigned int binxmax,
				  const unsigned int nbin, const std::string binning_option,
				  const int range, const double fitmin, const double fitmax,
				  const std::string fit_plot_name) const
{
  XSlice("Sigma", gr, binxmin, binxmax, nbin, binning_option, "Gauss", range, fitmin, fitmax, fit_plot_name);
}

void TH2Analyzer::MeanXSlice(TH1D* histo, const unsigned int binxmin, const unsigned int binxmax,
		 const unsigned int nbin, const std::string binning_option) const
{
  checkBinning();
  double *xlow=new double[nbin+1];
  binning_computation(xlow,nbin,binxmin,binxmax,binning_option);
  histo->SetBins(nbin, xlow);
  for (unsigned int binc=0;binc<nbin;++binc)
  {
    const int binxminc=static_cast<int>(xlow[binc]);
    const int binxmaxc=static_cast<int>(xlow[binc+1]);
    histo->SetBinContent(binc+1,binxminc+(binxmaxc-binxminc)/2.0);
    histo->SetBinError(binc+1,(binxmaxc-binxminc)/2.0);
  }
  delete [] xlow;
}

void TH2Analyzer::binning_computation(double* xlow, const unsigned int nbin, const unsigned int binxmin,
			    const unsigned int binxmax, const std::string binning_option) const
{
  if (binning_option=="var")
  {
    // (we want approx. the same number of entries per bin)
    TH1D* h0_slice1 = this->ProjectionY("h0_slice1",binxmin, binxmax, "");
    const unsigned int totalNumberOfEvents=static_cast<unsigned int>(h0_slice1->GetEntries());
    //std::cout << "totalNumberOfEvents = " << totalNumberOfEvents << std::endl;
    delete h0_slice1;

    unsigned int neventsc=0;
    unsigned int binXmaxc=binxmin+1;
    xlow[0]=binxmin;
    for (unsigned int binc=1;binc<nbin;++binc)
    {
      while (static_cast<double>(neventsc)<(binc)*totalNumberOfEvents/nbin)
      {
        TH1D* h0_slice1c = this->ProjectionY("h0_slice1",binxmin, binXmaxc, "");
        neventsc=static_cast<unsigned int>(h0_slice1c->GetEntries());
        //std::cout << "FL : neventsc = " << neventsc << std::endl;
        //std::cout << "FL : binXmaxc = " << binXmaxc << std::endl;
        ++binXmaxc;
        delete h0_slice1c;
      }
      //std::cout << "binXmaxc-1 = " << binXmaxc-1 << std::endl;
      xlow[binc]=binXmaxc-1;
    }
    xlow[nbin]=binxmax;
  }
  else
  {
    if (binning_option!="cst")
    {
      std::cout << "Error: binning_option = " << binning_option << "  unknown !" << std::endl;
      std::cout << "cst option will be used" << std::endl;
    }
    for (unsigned int binc=0;binc<nbin+1;++binc)
    {
      xlow[binc]=binxmin+binc*(binxmax-binxmin)/nbin;
    }
  }
}
