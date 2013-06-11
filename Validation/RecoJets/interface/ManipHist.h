#ifndef ManipHist_h
#define ManipHist_h

#include "Validation/RecoJets/interface/FitHist.h"

class ManipHist : public FitHist{

 public:

  ManipHist(){};
  //~ManipHist(){ file_->Close(); };
  virtual ~ManipHist(){};

  //extra members
  void sumHistograms();
  void divideAndDrawPs();
  void divideAndDrawEps();
  
 protected:

  //specific configurables
  void configBlockSum(ConfigFile&);
  void configBlockDivide(ConfigFile&);
  
  //extra members
  TH1F& divideHistograms(TH1F&, TH1F&, int); 
  double ratioCorrelatedError(double&, double&, double&, double&);
  double ratioUncorrelatedError(double&, double&, double&, double&);
  
 protected:

  //---------------------------------------------
  // Interface
  //---------------------------------------------

  //define histogram manipulations
  int errorType_;                          // define histogram errors (uncorr/corr)
  std::vector<double> weights_;            // define weights
};

#endif
