#ifndef FitHist_h
#define FitHist_h

#include "Validation/RecoJets/interface/CompHist.h"
#include "Validation/RecoJets/interface/CompMethods.h"


class FitHist : public CompHist{

 public:

  enum EvalType{kStabilizedGauss, kHistogramMean, kMaximalValue, kQuantile};

 public:

  FitHist():CompHist(false){};
  FitHist(bool verbose):CompHist(verbose){};
  //~FitHist(){ file_->Close(); };
  virtual ~FitHist(){};

  //extra members
  void fitAndDrawPs();
  void fitAndDrawEps();
  void fillTargetHistograms();
  void writeFitOutput();

 protected:

  //io helpers
  void configBlockFit(ConfigFile&);

  //extra members
  bool checkTargetHistList();
  bool isInFitTargetList(std::string&);
  TH1F& findFitHistogram(const TObjArray&, TString&, TString&, int&); 
  TH1F& findTargetHistogram(const TObjArray&, TString&, TString&, TString&);
  double normalize(TString&, double); 
  void fillTargetHistogram(std::string&);
  void fillTargetHistogramBin(TH1F&, TH1F&, int);
  void fillTargetHistogramBin(TH1F&, TH1F&, int, TString&, Quantile&);
  void fillTargetHistogramBin(TH1F&, TH1F&, int, TString&, MaximalValue&);
  void fillTargetHistogramBin(TH1F&, TH1F&, int, TString&, HistogramMean&);
  void fillTargetHistogramBin(TH1F&, TH1F&, int, TString&, StabilizedGauss&);
  void setFitHistogramAxes(TH1F&, int);
  void addBinLabelToFitHist(const TObjArray&, int&, TString&, TString&);
  void addParLabelToFitHist(const TH1F&);

 protected:

  //---------------------------------------------
  // Interface
  //---------------------------------------------

  //define additional input
  std::vector<std::string> targetHistList_;  // list of target histograms to be filled

  //define additional histogram design
  std::vector<int> axesIndex_;               // index list of axis titels of fit histograms
  std::vector<std::string> xAxesFit_;        // x axis title(s) of fit histograms 
  std::vector<std::string> yAxesFit_;        // y axis title(s) of fit histograms

  //define fit/eval procedure

  int evalType_;                             // evaluation type
  int fitFuncType_;                          // fit function type
  std::string fitFuncName_;                  // fit functino name
  std::string fitFuncTitle_;                 // fit function title (to be shown in legend)
  double fitFuncLowerBound_;                 // lower bound of fit function
  double fitFuncUpperBound_;                 // upper bound of fit function
};

#endif
