#ifndef PLOT_COMPARE_UTILITY__H
#define PLOT_COMPARE_UTILITY__H

#include "HistoData.h"

#include <map>
#include <string>
#include <vector>

#ifndef HTML1D
#define HTML1D
#endif

class TFile;
class TH1;

class PlotCompareUtility {
public:
  // BasePath = the path to data in the DQM root file (eg.,
  // "DQMData/METTask/ECAL/data") Prefix = the prefix common to all histograms
  // in area (eg., "METTask_" or "")
  PlotCompareUtility(std::string Reference,
                     std::string New,
                     std::string NewBasePath,
                     std::string NewPrefix = "",
                     std::string RefBasePath = "",
                     std::string RefPrefix = "");
  virtual ~PlotCompareUtility();

  // Axis Conventions (For Specifying Profiles, Projections, etc.)
  enum Axis { axisX, axisY /*, axisZ -- maybe later? */ };

  // Getters for HistoData Information
  std::vector<HistoData> *getHistos() { return &histos; }
  std::vector<HistoData> *getProjectionsX(HistoData *HD) { return &projectionsX[HD]; }
  std::vector<HistoData> *getProjectionsY(HistoData *HD) { return &projectionsY[HD]; }
  int getNumHistos() const { return histos.size(); }
  // int getNumProjectionsX(HistoData *HD) const { return
  // projectionsX[HD].size(); } int getNumProjectionsY(HistoData *HD) const {
  // return projectionsY[HD].size(); }

  // Getters for Statistical Comparisons
  double getKSThreshold() const { return ksThreshold; }
  double getChi2Threshold() const { return chi2Threshold; }
  double getThreshold() const;  // the lowest non-zero test threshold
  bool getFinalResult() const { return finalResult; }

  // Getters for Drawing Options
  int getSummaryWidth() const { return summaryWidth; }
  int getSummaryHeight() const { return summaryHeight; }
  int getSummaryBarsThickness() const { return summaryBarsThickness; }
  int getSummaryTopMargin() const { return summaryTopMargin; }
  int getSummaryLeftMargin() const { return summaryLeftMargin; }
  int getSummaryRightMargin() const { return summaryRightMargin; }
  int getSummaryBottomMargin() const { return summaryBottomMargin; }
  int getProjectionsHeight() const { return projectionsHeight; }
  int getProjectionsWidth() const { return projectionsWidth; }
  int getProjectionsBarsThickness() const { return projectionsBarsThickness; }
  int getProjectionsTopMargin() const { return projectionsTopMargin; }
  int getProjectionsLeftMargin() const { return projectionsLeftMargin; }
  int getProjectionsRightMargin() const { return projectionsRightMargin; }
  int getProjectionsBottomMargin() const { return projectionsBottomMargin; }
  int getPlotsHeight() const { return plotsHeight; }
  int getPlotsWidth() const { return plotsWidth; }
  int getPlotsTopMargin() const { return plotsTopMargin; }
  int getPlotsLeftMargin() const { return plotsLeftMargin; }
  int getPlotsRightMargin() const { return plotsRightMargin; }
  int getPlotsBottomMargin() const { return plotsBottomMargin; }

  // Setters for Statistical Comparisons
  void setKSThreshold(double Threshold) { ksThreshold = Threshold; }
  void setChi2Threshold(double Threshold) { chi2Threshold = Threshold; }

  // Setters for Drawing Options
  void setSummaryWidth(int Pixels) { summaryWidth = Pixels; }
  void setSummaryHeight(int Pixels) { summaryHeight = Pixels; }
  void setSummaryBarsThickness(int Pixels) { summaryBarsThickness = Pixels; }
  void setSummaryTopMargin(int Pixels) { summaryTopMargin = Pixels; }
  void setSummaryLeftMargin(int Pixels) { summaryLeftMargin = Pixels; }
  void setSummaryRightMargin(int Pixels) { summaryRightMargin = Pixels; }
  void setSummaryBottomMargin(int Pixels) { summaryBottomMargin = Pixels; }
  void setProjectionsiWidth(int Pixels) { projectionsWidth = Pixels; }
  void setProjectionsHeight(int Pixels) { projectionsHeight = Pixels; }
  void setProjectionsBarsThickness(int Pixels) { projectionsBarsThickness = Pixels; }
  void setProjectionsTopMargin(int Pixels) { projectionsTopMargin = Pixels; }
  void setProjectionsLeftMargin(int Pixels) { projectionsLeftMargin = Pixels; }
  void setProjectionsRightMargin(int Pixels) { projectionsRightMargin = Pixels; }
  void setProjectionsBottomMargin(int Pixels) { projectionsBottomMargin = Pixels; }
  void setPlotsHeight(int Pixels) { plotsHeight = Pixels; }
  void setPlotsWidth(int Pixels) { plotsWidth = Pixels; }
  void setPlotsTopMargin(int Pixels) { plotsTopMargin = Pixels; }
  void setPlotsLeftMargin(int Pixels) { plotsLeftMargin = Pixels; }
  void setPlotsRightMargin(int Pixels) { plotsRightMargin = Pixels; }
  void setPlotsBottomMargin(int Pixels) { plotsBottomMargin = Pixels; }

  // Add HistoData Objects for Comparison
  HistoData *addHistoData(std::string NewName, std::string RefName, int PlotType);
  HistoData *addHistoData(std::string Name, int PlotType) { return addHistoData(Name, Name, PlotType); }
  HistoData *addProjectionXData(
      HistoData *Parent, std::string Name, int PlotType, int Bin, TH1 *NewHisto, TH1 *RefHisto);
  HistoData *addProjectionYData(
      HistoData *Parent, std::string Name, int PlotType, int Bin, TH1 *NewHisto, TH1 *RefHisto);
  void clearHistos() { histos.clear(); }
  void clearProjectionsX(HistoData *Parent) { projectionsX[Parent].clear(); }
  void clearProjectionsY(HistoData *Parent) { projectionsY[Parent].clear(); }

  // Misc. Utilities
  bool compare(HistoData *);
  void makeDefaultPlots();
  void makePlots(HistoData *);
  void makeHTML(HistoData *);
  void makeSummary(std::string Name);
  void makeSummaryPlot(std::string Name);
  void makeSummaryHTML(std::string Name);
  // void makeProjectionsSummary();
  // void makeProjectionsSummaryPlots();
  // void makeProjectionsSummaryHTML();
  bool isValid() const;
  void dump();

private:
  // data holders for histogram types
  std::vector<HistoData> histos;
  std::map<HistoData *, std::vector<HistoData>> projectionsX;
  std::map<HistoData *, std::vector<HistoData>> projectionsY;

  // file pointers and file organization
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
  template <int PlotType>
  bool compare(HistoData *);
  template <int PlotType>
  void makePlots(HistoData *);
  template <int PlotType>
  void makeHTML(HistoData *);
  void centerRebin(TH1 *, TH1 *);
  void renormalize(TH1 *, TH1 *);

  // summary settings
  int summaryWidth;   // user defined
  int summaryHeight;  // set by plotter
  int summaryBarsThickness;
  int summaryTopMargin;
  int summaryLeftMargin;
  int summaryRightMargin;
  int summaryBottomMargin;

  // 2d projections summary settings
  int projectionsWidth;   // set by plotter
  int projectionsHeight;  // user defined
  int projectionsBarsThickness;
  int projectionsTopMargin;
  int projectionsLeftMargin;
  int projectionsRightMargin;
  int projectionsBottomMargin;

  // 1d distribution plots settings
  int plotsWidth;   // user defined
  int plotsHeight;  // user defined
  int plotsTopMargin;
  int plotsLeftMargin;
  int plotsRightMargin;
  int plotsBottomMargin;

  // true if all run tests pass, false if any fail
  bool finalResult;
};

#endif  // PLOT_COMPARE_UTILITY__H
