#ifndef PLOT_COMPARE_UTILITY__H
#define PLOT_COMPARE_UTILITY__H

#include <vector>
#include <string>

#include "HistoData.h"

class TStyle;
class TFile;
class TObject;

TStyle tdrstyle();

class PlotCompareUtility {
public:

  PlotCompareUtility(std::string Reference, std::string New, std::string DataPath, std::string Prefix = "");
  virtual ~PlotCompareUtility();

  double Compare(TH1F*, TH1F*, int);
  
  // Setters
  void SetKSThreshold(float v) { ks_threshold = v; } 
  void SetChi2Threshold(float v) { chi2_threshold = v; } 
  void SetDataPath(std::string v) { base_path = v; }
  void SetStyle(TStyle v) { style = v; }
  void SetPageTitle(std::string v) { comp_title = v; }

  // Getters
  TFile *GetNewFile() { return new_file; }
  TFile *GetRefFile() { return ref_file; }
  TObject *GetNewHisto(std::string Name);
  TObject *GetRefHisto(std::string Name);
  float GetKSThreshold() { return ks_threshold; }
  float GetChi2Threshold() { return chi2_threshold; }
  std::string GetDataPath() { return base_path; }
  TStyle GetStyle() { return style; }
  std::string GetPageTitle() { return comp_title; }
  int GetStatus() { return status; }
  std::string GetPrefix() { return prefix; }

  // Histogram Data Accessors
  int GetNumHistos();
  HistoData *AddHistoData(std::string Name);
  HistoData *GetHistoData(std::string Name);
  HistoData *GetHistoData(int);

private:

  // Histogram Data Holder
  std::vector<HistoData *> histo_d;

  // Validation/RecoMET Output Files
  TFile *ref_file;
  TFile *new_file;

  // The Minimum Passing KS/Chi2 Score
  float ks_threshold;
  float chi2_threshold;

  // The Path to Data in the DQM Root File (eg., DQMData/METTask/ECAL/data)
  std::string base_path;

  // the prefix common to all histograms in area (eg., METTask_)
  std::string prefix;

  // Main Axis Title (eg., Compatibility or Compatibility Range ...)
  std::string comp_title;

  // Data Holder for the TDR Style (Use the "tdrstyle()" Function)
  TStyle style;

  // Keep track of errors (throw implementation was buggy)
  int status;

};

#endif // PLOT_COMPARE_UTILITY__H
