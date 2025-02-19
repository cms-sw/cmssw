#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>

#include "Validation/RecoJets/interface/CompMethods.h"

using std::cerr;
using std::cout;
using std::endl;

double Gauss(double *x, double *par){
  double arg=0;
  if(par[2]!=0){
    arg = (x[0]-par[1])/par[2];
  }
  return par[0]*TMath::Exp(-0.5*arg*arg);
}

// ------------------------------------------------------------------------------------------------------------------

StabilizedGauss::StabilizedGauss(const char* funcName, int funcType, double lowerBound, double upperBound):
  funcType_  (funcType  ),
  funcName_  (funcName  ),
  lowerBound_(lowerBound),
  upperBound_(upperBound)
{
  if(funcType_==0){
    func_ = new TF1(funcName_, Gauss, lowerBound_, upperBound_, 3);  
    func_->SetParNames( "Const", "Mean", "Sigma" );
  }
  else{
    std::cout << "Sorry: not yet implemented" << std::endl;
  }
}

void
StabilizedGauss::fit(TH1F& hist)
{
  //set start values for first iteration
  if(funcType_==0){
    double maxValue=hist.GetBinCenter(hist.GetMaximumBin());
    func_->SetParameter(1, maxValue);
    func_->SetParameter(2, hist.GetRMS());
  }

  //set parameter limits
  if(funcType_==0){
    func_->SetParLimits(1, lowerBound_, upperBound_);
    func_->SetParLimits(2, 0., 5.*hist.GetRMS());
  }

  //do the fit
  mean_ = func_->GetParameter(1);
  sigma_= func_->GetParameter(2);

  hist.Fit( "func", "RE0", "", (mean_-2.*sigma_), (mean_+2.*sigma_) );
  if(hist.GetFunction("func")){
    //get mean and sigma 
    //from first iteration
    mean_ = hist.GetFunction("func")->GetParameter(1);
    sigma_= hist.GetFunction("func")->GetParameter(2);

    //set start values for 
    //second iteration
    func_->SetParameter(1, mean_ );
    func_->SetParameter(2, sigma_);
    hist.Fit( func_, "MEL", "", (mean_-1.5*sigma_), (mean_+1.5*sigma_) );
  }
  else{
    std::cout << "sorry... no fit function found..." << std::endl;
  }
}

// ------------------------------------------------------------------------------------------------------------------

std::pair<int,int>
MaximalValue::contour(TH1F& hist, double& frac)
{
  int idx=hist.GetMaximumBin(), jdx=hist.GetMaximumBin();
  if(0<=frac && frac<=1){
    while( hist.GetBinContent(idx)/hist.GetMaximum()>frac) --idx;
    while( hist.GetBinContent(jdx)/hist.GetMaximum()>frac) ++jdx;
  }
  return std::pair<int, int>(idx, jdx);
}

// ------------------------------------------------------------------------------------------------------------------

double 
Quantile::spreadError(TH1F& hist)
{
  quantiles(hist, 0.25+err_); double outer=distance(hist);
  quantiles(hist, 0.25-err_); double inner=distance(hist);
  return std::fabs(outer-inner)/2;
}
