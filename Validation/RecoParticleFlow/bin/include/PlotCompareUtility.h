#ifndef PLOT_COMPARE_UTILITY__H
#define PLOT_COMPARE_UTILITY__H

#include "HistoData.h"

#include <vector>
#include <map>
#include <string>

class TFile;
class TH1;

class PlotCompareUtility {
public:

  // BasePath = the path to data in the DQM root file (eg., "DQMData/METTask/ECAL/data")
  // Prefix = the prefix common to all histograms in area (eg., "METTask_" or "")
  PlotCompareUtility(std::string Reference, std::string New, std::string NewBasePath,
    std::string NewPrefix = "", std::string RefBasePath = "", std::string RefPrefix = "");
  virtual ~PlotCompareUtility();

  // Axis Conventions (For Specifying Profiles, Projections, etc.)
  enum Axis { axisX, axisY /*, axisZ -- maybe later? */ };

  // Getters for HistoData Information
  std::vector<HistoData> *getHistos() { return &histos; }
  std::vector<HistoData> *getProjectionsX(HistoData *HD) { return &projectionsX[HD]; }
  std::vector<HistoData> *getProjectionsY(HistoData *HD) { return &projectionsY[HD]; }
  unsigned int getNumHistos() const { return histos.size(); } 
  //unsigned short getNumProjectionsX(HistoData *HD) const { return projectionsX[HD].size(); }
  //unsigned short getNumProjectionsY(HistoData *HD) const { return projectionsY[HD].size(); }

  // Getters for Statistical Comparisons
  double getKSThreshold() const { return ksThreshold; }
  double getChi2Threshold() const { return chi2Threshold; }
  double getThreshold() const; // the lowest non-zero test threshold
  bool getFinalResult() const { return finalResult; }

  // Getters for Drawing Options
  unsigned short getSummaryWidth() const { return summaryWidth; }
  unsigned short getSummaryHeight() const { return summaryHeight; }
  unsigned short getSummaryBarsThickness() const { return summaryBarsThickness; }
  unsigned short getSummaryTopMargin() const { return summaryTopMargin; }
  unsigned short getSummaryLeftMargin() const { return summaryLeftMargin; }
  unsigned short getSummaryRightMargin() const { return summaryRightMargin; }
  unsigned short getSummaryBottomMargin() const { return summaryBottomMargin; }

  // Setters for Statistical Comparisons
  void setKSThreshold(double Threshold) { ksThreshold = Threshold; }
  void setChi2Threshold(double Threshold) { chi2Threshold = Threshold; }

  // Setters for Summary Drawing Options
  void setSummaryWidth(unsigned short Pixels) { summaryWidth = Pixels; }
  void setSummaryHeight(unsigned short Pixels) { summaryHeight = Pixels; }
  void setSummaryBarsThickness(unsigned short Pixels) { summaryBarsThickness = Pixels; }
  void setSummaryTopMargin(unsigned short Pixels) { summaryTopMargin = Pixels; }
  void setSummaryLeftMargin(unsigned short Pixels) { summaryLeftMargin = Pixels; }
  void setSummaryRightMargin(unsigned short Pixels) { summaryRightMargin = Pixels; }
  void setSummaryBottomMargin(unsigned short Pixels) { summaryBottomMargin = Pixels; }

  // Add HistoData Objects for Comparison
  HistoData *addHistoData(std::string NewName, std::string RefName, int PlotType);
  HistoData *addHistoData(std::string Name, int PlotType) { return addHistoData(Name,Name,PlotType); }
  HistoData *addProjectionXData(HistoData *Parent, std::string Name, int PlotType, int Bin, TH1* NewHisto, TH1* RefHisto);
  HistoData *addProjectionYData(HistoData *Parent, std::string Name, int PlotType, int Bin, TH1* NewHisto, TH1* RefHisto);

  // Misc. Utilities
  bool compare(HistoData *);
  void makeDefaultPlots();
  void makePlots(HistoData *);
  void makeHTML(HistoData *);
  void makeSummary(std::string Name);
  void makeSummaryPlot(std::string Name);
  void makeSummaryHTML(std::string Name);
  bool isValid() const;
  void dump();

private:

  // data holders for histograms
  std::vector<HistoData> histos;
  std::map<HistoData *,std::vector<HistoData> > projectionsX;
  std::map<HistoData *,std::vector<HistoData> > projectionsY;

  // file pounsigned shorters and file organization
  TFile *refFile;
  TFile *newFile;
  std::string newBasePath;
  std::string newPrefix;
  std::string refBasePath;
  std::string refPrefix;

  // statistical thresholds
  double ksThreshold;
  double chi2Threshold;

  // private (implementation/helper) functions
  template <int PlotType> bool compare(HistoData *);
  template <int PlotType> void makePlots(HistoData *);
  template <int PlotType> void makeHTML(HistoData *);
  void centerRebin(TH1 *, TH1 *);
  void renormalize(TH1 *, TH1 *);

  // summary settings
  unsigned short summaryWidth; // user defined
  unsigned short summaryHeight; // set by plotter
  unsigned short summaryBarsThickness;
  unsigned short summaryTopMargin;
  unsigned short summaryLeftMargin;
  unsigned short summaryRightMargin;
  unsigned short summaryBottomMargin;

  // default image filenames
  std::string noDataImage;

  // true if all run tests pass, false if any fail
  bool finalResult;

};

#endif // PLOT_COMPARE_UTILITY__H
