#ifndef CompMethods_h
#define CompMethods_h

#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>

#include "Validation/RecoJets/interface/RootSystem.h"
#include "Validation/RecoJets/interface/RootHistograms.h"


class MaximalValue {

 public:

  MaximalValue(double val, double err):val_(val), err_(err){};
  ~MaximalValue(){};
  double value(TH1F& hist){ return hist.GetBinCenter(hist.GetMaximumBin()); };
  double valueError(TH1F& hist) { return spread(hist, val_); };
  double spread(TH1F& hist) { return spread(hist, 0.5); };
  double spreadError(TH1F& hist){ return std::fabs(spread(hist, 0.5-err_)-spread(hist, 0.5+err_))/2; };
  
 private:

  std::pair<int,int> contour(TH1F&, double&);
  double spread(TH1F& hist, double frac){ return std::fabs(hist.GetBinCenter(contour(hist, frac).second)-hist.GetBinCenter(contour(hist, frac).first)); };

 private:

  double val_;
  double err_;
};

// ------------------------------------------------------------------------------------------------------------------

class Quantile {

 public:

  Quantile(double central, double err):central_(central), err_(err){};
  ~Quantile(){};
  double value(TH1F& hist){ quantiles(hist, err_); return qnt_[1]; };
  double valueError(TH1F& hist) { quantiles(hist, err_); return distance(hist); };
  double spread(TH1F& hist) { quantiles(hist, 0.25); return distance(hist); };
  double spreadError(TH1F& hist);

 private:
  
  void evaluate(double& err){val_[0]=central_-err; val_[1]=central_; val_[2]=central_+err;};
  void quantiles(TH1F& hist, double err){ evaluate(err); hist.GetQuantiles(3, qnt_, val_); };
  double distance(TH1F& hist){ return std::fabs(qnt_[2]-qnt_[0]); };
  
 private:

  double central_;
  double err_;
  double val_[3];
  double qnt_[3];
};

// ------------------------------------------------------------------------------------------------------------------

class HistogramMean {

 public:

  HistogramMean(){};
  ~HistogramMean(){};
  double value(TH1F& hist){ return hist.GetMean(); };
  double valueError(TH1F& hist) { return hist.GetMeanError(); };
  double spread(TH1F& hist) { return TMath::Sqrt(hist.GetRMS()); };
  double spreadError(TH1F& hist){ return hist.GetRMSError()/TMath::Sqrt(hist.GetRMS()); };
};

// ------------------------------------------------------------------------------------------------------------------

class StabilizedGauss {

 public:

  StabilizedGauss(){ func_=new TF1(); };
  StabilizedGauss(const char* name):funcName_(name){ func_=new TF1(); };
  StabilizedGauss(const char*, int, double, double);
  ~StabilizedGauss(){ delete func_;};
  void fit(TH1F&);
  double value(TH1F& hist){ return value(hist, 1); };
  double valueError(TH1F& hist){ return error(hist, 1); };
  double spread(TH1F& hist){ return value(hist, 2); };
  double spreadError(TH1F& hist){ return error(hist, 2); };

 private:

  double value(TH1F& hist, int target){ return (hist.GetFunction( funcName_) ? hist.GetFunction( funcName_)->GetParameter(target) : 0.); }
  double error(TH1F& hist, int target){ return (hist.GetFunction( funcName_) ? hist.GetFunction( funcName_)->GetParError (target) : 0.); }

 private:

  TF1* func_;
  double mean_;
  double sigma_;
  int funcType_;
  const char* funcName_;
  double lowerBound_;
  double upperBound_;
};

#endif
