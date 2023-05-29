#include "Validation/RecoTau/plugins/DQMHistPlotter.h"

#include "Validation/RecoTau/plugins/dqmAuxFunctions.h"

// framework & common header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TCanvas.h>
#include <TPad.h>
#include <TPostScript.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TMath.h>

#include <iostream>

namespace {
  //defaults for cfgEntryProcess
  const std::string type_smMC = "smMC";
  const std::string type_bsmMC = "bsmMC";
  const std::string type_smSumMC = "smSumMC";
  const std::string type_Data = "Data";

  //defaults for cfgEntryAxisX
  const double defaultMinX = -1.;
  const double defaultMaxX = -1.;
  const double defaultXaxisTitleOffset = 1.0;
  const double defaultXaxisTitleSize = 0.05;

  //defaults for cfgEntryAxisY
  const double defaultMinY_linear = 0.;
  const double defaultMinY_log = 1.e-2;
  const double defaultMaxY_linear = -1.;
  const double defaultMaxY_log = -1.;
  const std::string yScale_linear = "linear";
  const std::string yScale_log = "log";
  const std::string defaultYscale = yScale_linear;
  const double defaultYaxisTitleOffset = 1.0;
  const double defaultYaxisTitleSize = 0.05;
  const double defaultYaxisMaximumScaleFactor_linear = 1.6;
  const double defaultYaxisMaximumScaleFactor_log = 5.e+2;

  // defaults for cfgEntryLegend
  const double defaultLegendPosX = 0.50;
  const double defaultLegendPosY = 0.55;
  const double defaultLegendSizeX = 0.39;
  const double defaultLegendSizeY = 0.34;
  const std::string defaultLegendHeader = "";
  const std::string defaultLegendOptions = "brNDC";
  const int defaultLegendBorderSize = 0;
  const int defaultLegendFillColor = 0;

  // defaults for cfgEntryLabel
  const double defaultLabelPosX = 0.66;
  const double defaultLabelPosY = 0.82;
  const double defaultLabelSizeX = 0.26;
  const double defaultLabelSizeY = 0.10;
  const std::string defaultLabelOptions = "brNDC";
  const int defaultLabelBorderSize = 0;
  const int defaultLabelFillColor = 0;
  const int defaultLabelTextColor = 1;
  const double defaultLabelTextSize = 0.05;
  const int defaultLabelTextAlign = 22;  // horizontally and vertically centered, see documentation of TAttText
  const double defaultLabelTextAngle = 0.;

  // defaults for cfgEntryDrawOption
  const int defaultMarkerColor = 1;
  const int defaultMarkerSize = 1;
  const int defaultMarkerStyle = 2;
  const int defaultLineColor = 0;
  const int defaultLineStyle = 1;
  const int defaultLineWidth = 2;
  const int defaultFillColor = 0;
  const int defaultFillStyle = 1001;
  const std::string defaultDrawOption = "";
  const std::string defaultDrawOptionLegend = "lpf";

  const std::string drawOption_eBand = "eBand";

  // global defaults
  const int defaultCanvasSizeX = 800;
  const int defaultCanvasSizeY = 600;

  const std::string drawOptionSeparator = "#.#";

  const int verbosity = 0;

  template <class T>
  void checkCfgDef(const std::string& cfgEntryName,
                   std::map<std::string, T>& def,
                   int& errorFlag,
                   const std::string& defType,
                   const std::string& drawJobName) {
    if (def.find(cfgEntryName) == def.end()) {
      edm::LogError("checkCfgDef") << " " << defType << " = " << cfgEntryName
                                   << " undefined, needed by drawJob = " << drawJobName << " !!";
      errorFlag = 1;
    }
  }

  template <class T>
  void checkCfgDefs(const std::vector<std::string>& cfgEntryNames,
                    std::map<std::string, T>& def,
                    int& errorFlag,
                    const std::string& defType,
                    const std::string& drawJobName) {
    for (std::vector<std::string>::const_iterator cfgEntryName = cfgEntryNames.begin();
         cfgEntryName != cfgEntryNames.end();
         ++cfgEntryName) {
      checkCfgDef(*cfgEntryName, def, errorFlag, defType, drawJobName);
    }
  }

  template <class T>
  const T* findCfgDef(const std::string& cfgEntryName,
                      std::map<std::string, T>& def,
                      const std::string& defType,
                      const std::string& drawJobName) {
    typename std::map<std::string, T>::const_iterator it = def.find(cfgEntryName);
    if (it != def.end()) {
      return &(it->second);
    } else {
      edm::LogError("findCfgDef") << " " << defType << " = " << cfgEntryName
                                  << " undefined, needed by drawJob = " << drawJobName << " !!";
      return nullptr;
    }
  }

  //
  //-----------------------------------------------------------------------------------------------------------------------
  //

  typedef std::pair<TH1*, std::string> histoDrawEntry;

  void drawHistograms(const std::list<histoDrawEntry>& histogramList, bool& isFirstHistogram) {
    for (std::list<histoDrawEntry>::const_iterator it = histogramList.begin(); it != histogramList.end(); ++it) {
      std::string drawOption = (isFirstHistogram) ? it->second : std::string(it->second).append("same");
      it->first->Draw(drawOption.data());
      isFirstHistogram = false;
    }
  }

  //
  //-----------------------------------------------------------------------------------------------------------------------
  //

  bool find_vstring(const std::vector<std::string>& vs, const std::string& s) {
    for (std::vector<std::string>::const_iterator it = vs.begin(); it != vs.end(); ++it) {
      if ((*it) == s)
        return true;
    }
    return false;
  }
}  // namespace
//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMHistPlotter::cfgEntryProcess::cfgEntryProcess(const std::string& name, const edm::ParameterSet& cfg) {
  name_ = name;

  dqmDirectory_ = cfg.getParameter<std::string>("dqmDirectory");

  legendEntry_ = cfg.getParameter<std::string>("legendEntry");
  legendEntryErrorBand_ = (cfg.exists("legendEntryErrorBand")) ? cfg.getParameter<std::string>("legendEntryErrorBand")
                                                               : std::string(legendEntry_).append(" Uncertainty");

  type_ = cfg.getParameter<std::string>("type");

  if (verbosity)
    print();
}

void TauDQMHistPlotter::cfgEntryProcess::print() const {
  std::cout << "<TauDQMHistPlotter::cfgEntryProcess::print>:" << std::endl;
  std::cout << " name = " << name_ << std::endl;
  std::cout << " dqmDirectory = " << dqmDirectory_ << std::endl;
  std::cout << " legendEntry = " << legendEntry_ << std::endl;
  std::cout << " legendEntryErrorBand = " << legendEntryErrorBand_ << std::endl;
  std::cout << " type = " << type_ << std::endl;
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMHistPlotter::cfgEntryAxisX::cfgEntryAxisX(const std::string& name, const edm::ParameterSet& cfg) {
  name_ = name;

  minX_ = (cfg.exists("minX")) ? cfg.getParameter<double>("minX") : defaultMinX;
  maxX_ = (cfg.exists("maxX")) ? cfg.getParameter<double>("maxX") : defaultMaxX;
  xAxisTitle_ = cfg.getParameter<std::string>("xAxisTitle");
  xAxisTitleOffset_ =
      (cfg.exists("xAxisTitleOffset")) ? cfg.getParameter<double>("xAxisTitleOffset") : defaultXaxisTitleOffset;
  xAxisTitleSize_ = (cfg.exists("xAxisTitleSize")) ? cfg.getParameter<double>("xAxisTitleSize") : defaultXaxisTitleSize;

  if (verbosity)
    print();
}

void TauDQMHistPlotter::cfgEntryAxisX::print() const {
  std::cout << "<TauDQMHistPlotter::cfgEntryAxisX::print>:" << std::endl;
  std::cout << " name = " << name_ << std::endl;
  std::cout << " minX_ = " << minX_ << std::endl;
  std::cout << " maxX_ = " << maxX_ << std::endl;
  std::cout << " xAxisTitle = " << xAxisTitle_ << std::endl;
  std::cout << " xAxisTitleOffset = " << xAxisTitleOffset_ << std::endl;
  std::cout << " xAxisTitleSize = " << xAxisTitleSize_ << std::endl;
}

void TauDQMHistPlotter::cfgEntryAxisX::applyTo(TH1* histogram) const {
  if (histogram) {
    double xMin = (minX_ != defaultMinX) ? minX_ : histogram->GetXaxis()->GetXmin();
    double xMax = (maxX_ != defaultMaxX) ? maxX_ : histogram->GetXaxis()->GetXmax();
    histogram->SetAxisRange(xMin, xMax, "X");
    histogram->GetXaxis()->SetTitle(xAxisTitle_.data());
    histogram->GetXaxis()->SetTitleOffset(xAxisTitleOffset_);
    histogram->GetXaxis()->SetTitleSize(xAxisTitleSize_);
  }
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMHistPlotter::cfgEntryAxisY::cfgEntryAxisY(const std::string& name, const edm::ParameterSet& cfg) {
  name_ = name;

  minY_linear_ = (cfg.exists("minY_linear")) ? cfg.getParameter<double>("minY_linear") : defaultMinY_linear;
  minY_log_ = (cfg.exists("minY_log")) ? cfg.getParameter<double>("minY_log") : defaultMinY_log;
  maxY_linear_ = (cfg.exists("maxY_linear")) ? cfg.getParameter<double>("maxY_linear") : defaultMaxY_linear;
  maxY_log_ = (cfg.exists("maxY_log")) ? cfg.getParameter<double>("maxY_log") : defaultMaxY_log;
  yScale_ = (cfg.exists("yScale")) ? cfg.getParameter<std::string>("yScale") : defaultYscale;
  yAxisTitle_ = cfg.getParameter<std::string>("yAxisTitle");
  yAxisTitleOffset_ =
      (cfg.exists("yAxisTitleOffset")) ? cfg.getParameter<double>("yAxisTitleOffset") : defaultYaxisTitleOffset;
  yAxisTitleSize_ = (cfg.exists("yAxisTitleSize")) ? cfg.getParameter<double>("yAxisTitleSize") : defaultYaxisTitleSize;

  if (verbosity)
    print();
}

void TauDQMHistPlotter::cfgEntryAxisY::print() const {
  std::cout << "<TauDQMHistPlotter::cfgEntryAxisY::print>:" << std::endl;
  std::cout << " name = " << name_ << std::endl;
  std::cout << " minY_linear = " << minY_linear_ << std::endl;
  std::cout << " minY_log = " << minY_log_ << std::endl;
  std::cout << " maxY_linear = " << maxY_linear_ << std::endl;
  std::cout << " maxY_log = " << maxY_log_ << std::endl;
  std::cout << " yScale = " << yScale_ << std::endl;
  std::cout << " yAxisTitle = " << yAxisTitle_ << std::endl;
  std::cout << " yAxisTitleOffset = " << yAxisTitleOffset_ << std::endl;
  std::cout << " yAxisTitleSize = " << yAxisTitleSize_ << std::endl;
}

void TauDQMHistPlotter::cfgEntryAxisY::applyTo(TH1* histogram, double norm) const {
  if (histogram) {
    bool yLogScale = (yScale_ == yScale_log) ? true : false;
    double minY = (yLogScale) ? minY_log_ : minY_linear_;
    histogram->SetMinimum(minY);
    double maxY = (yLogScale) ? maxY_log_ : maxY_linear_;
    double defaultMaxY = (yLogScale) ? defaultMaxY_log : defaultMaxY_linear;
    if (maxY != defaultMaxY) {
      //--- normalize y-axis range using given configuration parameter
      histogram->SetMaximum(maxY);
    } else {
      //--- in case configuration parameter for y-axis range not explicitely given,
      //    normalize y-axis range to maximum of any histogram included in drawJob
      //    times defaultYaxisMaximumScaleFactor (apply scale factor in order to make space for legend)
      double defaultYaxisMaximumScaleFactor =
          (yLogScale) ? defaultYaxisMaximumScaleFactor_log : defaultYaxisMaximumScaleFactor_linear;
      histogram->SetMaximum(defaultYaxisMaximumScaleFactor * norm);
    }
    histogram->GetYaxis()->SetTitle(yAxisTitle_.data());
    histogram->GetYaxis()->SetTitleOffset(yAxisTitleOffset_);
    histogram->GetYaxis()->SetTitleSize(yAxisTitleSize_);
  }
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMHistPlotter::cfgEntryLegend::cfgEntryLegend(const std::string& name, const edm::ParameterSet& cfg) {
  name_ = name;

  posX_ = (cfg.exists("posX")) ? cfg.getParameter<double>("posX") : defaultLegendPosX;
  posY_ = (cfg.exists("posY")) ? cfg.getParameter<double>("posY") : defaultLegendPosY;
  sizeX_ = (cfg.exists("sizeX")) ? cfg.getParameter<double>("sizeX") : defaultLegendSizeX;
  sizeY_ = (cfg.exists("sizeY")) ? cfg.getParameter<double>("sizeY") : defaultLegendSizeY;
  header_ = (cfg.exists("header")) ? cfg.getParameter<std::string>("header") : defaultLegendHeader;
  option_ = (cfg.exists("option")) ? cfg.getParameter<std::string>("option") : defaultLegendOptions;
  borderSize_ = (cfg.exists("borderSize")) ? cfg.getParameter<int>("borderSize") : defaultLegendBorderSize;
  fillColor_ = (cfg.exists("fillColor")) ? cfg.getParameter<int>("fillColor") : defaultLegendFillColor;

  if (verbosity)
    print();
}

void TauDQMHistPlotter::cfgEntryLegend::print() const {
  std::cout << "<TauDQMHistPlotter::cfgEntryLegend::print>:" << std::endl;
  std::cout << " name = " << name_ << std::endl;
  std::cout << " posX = " << posX_ << std::endl;
  std::cout << " posY = " << posY_ << std::endl;
  std::cout << " sizeX = " << sizeX_ << std::endl;
  std::cout << " sizeY = " << sizeY_ << std::endl;
  std::cout << " header = " << header_ << std::endl;
  std::cout << " option = " << option_ << std::endl;
  std::cout << " borderSize = " << borderSize_ << std::endl;
  std::cout << " fillColor = " << fillColor_ << std::endl;
}

void TauDQMHistPlotter::cfgEntryLegend::applyTo(TLegend* legend) const {
  if (legend) {
    legend->SetX1(posX_);
    legend->SetY1(posY_);
    legend->SetX2(posX_ + sizeX_);
    legend->SetY2(posY_ + sizeY_);
    legend->SetHeader(header_.data());
    legend->SetOption(option_.data());
    legend->SetBorderSize(borderSize_);
    legend->SetFillColor(fillColor_);
  }
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMHistPlotter::cfgEntryLabel::cfgEntryLabel(const std::string& name, const edm::ParameterSet& cfg) {
  name_ = name;

  posX_ = (cfg.exists("posX")) ? cfg.getParameter<double>("posX") : defaultLabelPosX;
  posY_ = (cfg.exists("posY")) ? cfg.getParameter<double>("posY") : defaultLabelPosY;
  sizeX_ = (cfg.exists("sizeX")) ? cfg.getParameter<double>("sizeX") : defaultLabelSizeX;
  sizeY_ = (cfg.exists("sizeY")) ? cfg.getParameter<double>("sizeY") : defaultLabelSizeY;
  option_ = (cfg.exists("option")) ? cfg.getParameter<std::string>("option") : defaultLabelOptions;
  borderSize_ = (cfg.exists("borderSize")) ? cfg.getParameter<int>("borderSize") : defaultLabelBorderSize;
  fillColor_ = (cfg.exists("fillColor")) ? cfg.getParameter<int>("fillColor") : defaultLabelFillColor;
  textColor_ = (cfg.exists("textColor")) ? cfg.getParameter<int>("textColor") : defaultLabelTextColor;
  textSize_ = (cfg.exists("textSize")) ? cfg.getParameter<double>("textSize") : defaultLabelTextSize;
  textAlign_ = (cfg.exists("textAlign")) ? cfg.getParameter<int>("textAlign") : defaultLabelTextAlign;
  textAngle_ = (cfg.exists("textAngle")) ? cfg.getParameter<double>("textAngle") : defaultLabelTextAngle;
  text_ = cfg.getParameter<vstring>("text");

  if (verbosity)
    print();
}

void TauDQMHistPlotter::cfgEntryLabel::print() const {
  std::cout << "<TauDQMHistPlotter::cfgEntryLabel::print>:" << std::endl;
  std::cout << " name = " << name_ << std::endl;
  std::cout << " posX = " << posX_ << std::endl;
  std::cout << " posY = " << posY_ << std::endl;
  std::cout << " sizeX = " << sizeX_ << std::endl;
  std::cout << " sizeY = " << sizeY_ << std::endl;
  std::cout << " option = " << option_ << std::endl;
  std::cout << " borderSize = " << borderSize_ << std::endl;
  std::cout << " fillColor = " << fillColor_ << std::endl;
  std::cout << " textColor = " << textColor_ << std::endl;
  std::cout << " textSize = " << textSize_ << std::endl;
  std::cout << " textAlign = " << textAlign_ << std::endl;
  std::cout << " textAngle = " << textAngle_ << std::endl;
  std::cout << " text = " << format_vstring(text_) << std::endl;
}

void TauDQMHistPlotter::cfgEntryLabel::applyTo(TPaveText* label) const {
  if (label) {
    //--- WARNING: need to call TPaveText::SetX1NDC, **not** TPaveText::SetX1 !!
    //             (see documentation of base-class constructor
    //               TPave::TPave(Double_t, Double_t,Double_t, Double_t, Int_t, Option_t*)
    //              in TPave.cxx for details)
    label->SetX1NDC(posX_);
    label->SetY1NDC(posY_);
    label->SetX2NDC(posX_ + sizeX_);
    label->SetY2NDC(posY_ + sizeY_);
    label->SetOption(option_.data());
    label->SetBorderSize(borderSize_);
    label->SetFillColor(fillColor_);
    label->SetTextColor(textColor_);
    label->SetTextSize(textSize_);
    label->SetTextAlign(textAlign_);
    label->SetTextAngle(textAngle_);
    for (vstring::const_iterator line = text_.begin(); line != text_.end(); ++line) {
      label->AddText(line->data());
    }
  }
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMHistPlotter::cfgEntryDrawOption::cfgEntryDrawOption(const std::string& name, const edm::ParameterSet& cfg) {
  name_ = name;

  markerColor_ = (cfg.exists("markerColor")) ? cfg.getParameter<int>("markerColor") : defaultMarkerColor;
  markerSize_ = (cfg.exists("markerSize")) ? cfg.getParameter<double>("markerSize") : defaultMarkerSize;
  markerStyle_ = (cfg.exists("markerStyle")) ? cfg.getParameter<int>("markerStyle") : defaultMarkerStyle;

  lineColor_ = (cfg.exists("lineColor")) ? cfg.getParameter<int>("lineColor") : defaultLineColor;
  lineStyle_ = (cfg.exists("lineStyle")) ? cfg.getParameter<int>("lineStyle") : defaultLineStyle;
  lineWidth_ = (cfg.exists("lineWidth")) ? cfg.getParameter<int>("lineWidth") : defaultLineWidth;

  fillColor_ = (cfg.exists("fillColor")) ? cfg.getParameter<int>("fillColor") : defaultFillColor;
  fillStyle_ = (cfg.exists("fillStyle")) ? cfg.getParameter<int>("fillStyle") : defaultFillStyle;

  drawOption_ = (cfg.exists("drawOption")) ? cfg.getParameter<std::string>("drawOption") : defaultDrawOption;
  drawOptionLegend_ =
      (cfg.exists("drawOptionLegend")) ? cfg.getParameter<std::string>("drawOptionLegend") : defaultDrawOptionLegend;

  if (verbosity)
    print();
}

TauDQMHistPlotter::cfgEntryDrawOption::cfgEntryDrawOption(const std::string& name, const cfgEntryDrawOption& blueprint)
    : name_(name),
      markerColor_(blueprint.markerColor_),
      markerSize_(blueprint.markerSize_),
      markerStyle_(blueprint.markerStyle_),
      lineColor_(blueprint.lineColor_),
      lineStyle_(blueprint.lineStyle_),
      lineWidth_(blueprint.lineWidth_),
      fillColor_(blueprint.fillColor_),
      fillStyle_(blueprint.fillStyle_),
      drawOption_(blueprint.drawOption_),
      drawOptionLegend_(blueprint.drawOptionLegend_) {
  if (verbosity)
    print();
}

void TauDQMHistPlotter::cfgEntryDrawOption::print() const {
  std::cout << "<TauDQMHistPlotter::cfgEntryDrawOption::print>:" << std::endl;
  std::cout << " name = " << name_ << std::endl;
  std::cout << " markerColor = " << markerColor_ << std::endl;
  std::cout << " markerSize = " << markerSize_ << std::endl;
  std::cout << " markerStyle = " << markerStyle_ << std::endl;
  std::cout << " lineColor = " << lineColor_ << std::endl;
  std::cout << " lineStyle = " << lineStyle_ << std::endl;
  std::cout << " lineWidth = " << lineWidth_ << std::endl;
  std::cout << " fillColor = " << fillColor_ << std::endl;
  std::cout << " fillStyle = " << fillStyle_ << std::endl;
  std::cout << " drawOption = " << drawOption_ << std::endl;
  std::cout << " drawOptionLegend = " << drawOptionLegend_ << std::endl;
}

void TauDQMHistPlotter::cfgEntryDrawOption::applyTo(TH1* histogram) const {
  if (histogram) {
    histogram->SetMarkerColor(markerColor_);
    histogram->SetMarkerSize(markerSize_);
    histogram->SetMarkerStyle(markerStyle_);
    histogram->SetLineColor(lineColor_);
    histogram->SetLineStyle(lineStyle_);
    histogram->SetLineWidth(lineWidth_);
    histogram->SetFillColor(fillColor_);
    histogram->SetFillStyle(fillStyle_);
  }
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMHistPlotter::plotDefEntry::plotDefEntry(const std::string& dqmMonitorElement,
                                              const std::string& drawOptionEntry,
                                              const std::string& legendEntry,
                                              const std::string& legendEntryErrorBand,
                                              const std::string& process,
                                              bool doStack)
    : dqmMonitorElement_(dqmMonitorElement),
      drawOptionEntry_(drawOptionEntry),
      legendEntry_(legendEntry),
      legendEntryErrorBand_(legendEntryErrorBand),
      process_(process),
      doStack_(doStack),
      isErrorBand_(false) {
  //if ( verbosity ) print();
}

TauDQMHistPlotter::plotDefEntry::plotDefEntry(const plotDefEntry& blueprint)
    : dqmMonitorElement_(blueprint.dqmMonitorElement_),
      drawOptionEntry_(blueprint.drawOptionEntry_),
      legendEntry_(blueprint.legendEntry_),
      legendEntryErrorBand_(blueprint.legendEntryErrorBand_),
      process_(blueprint.process_),
      doStack_(blueprint.doStack_),
      isErrorBand_(false) {
  //if ( verbosity ) print();
}

void TauDQMHistPlotter::plotDefEntry::print() const {
  std::cout << "<TauDQMHistPlotter::plotDefEntry::print>:" << std::endl;
  std::cout << " dqmMonitorElement = " << dqmMonitorElement_ << std::endl;
  std::cout << " drawOptionEntry = " << drawOptionEntry_ << std::endl;
  std::cout << " legendEntry = " << legendEntry_ << std::endl;
  std::cout << " legendEntryErrorBand = " << legendEntryErrorBand_ << std::endl;
  std::cout << " process = " << process_ << std::endl;
  std::cout << " doStack = " << doStack_ << std::endl;
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMHistPlotter::cfgEntryDrawJob::cfgEntryDrawJob(const std::string& name,
                                                    const plotDefList& plotDefList,
                                                    const std::string& title,
                                                    const std::string& xAxis,
                                                    const std::string& yAxis,
                                                    const std::string& legend,
                                                    const vstring& labels) {
  name_ = name;

  for (plotDefList::const_iterator it = plotDefList.begin(); it != plotDefList.end(); ++it) {
    plots_.push_back(plotDefEntry(*it));
  }

  title_ = title;

  xAxis_ = xAxis;
  yAxis_ = yAxis;

  legend_ = legend;

  for (vstring::const_iterator it = labels.begin(); it != labels.end(); ++it) {
    labels_.push_back(std::string(*it));
  }

  if (verbosity)
    print();
}

void TauDQMHistPlotter::cfgEntryDrawJob::print() const {
  std::cout << "<TauDQMHistPlotter::cfgSetDrawJob::print>:" << std::endl;
  std::cout << " name = " << name_ << std::endl;
  std::cout << "plots = {" << std::endl;
  for (plotDefList::const_iterator plot = plots_.begin(); plot != plots_.end(); ++plot) {
    plot->print();
  }
  std::cout << "}" << std::endl;
  std::cout << " title = " << title_ << std::endl;
  std::cout << " xAxis = " << xAxis_ << std::endl;
  std::cout << " yAxis = " << yAxis_ << std::endl;
  std::cout << " legend = " << legend_ << std::endl;
  std::cout << " labels = " << format_vstring(labels_) << std::endl;
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMHistPlotter::TauDQMHistPlotter(const edm::ParameterSet& cfg) {
  usesResource("DQMStore");
  if (verbosity)
    std::cout << "<TauDQMHistPlotter::TauDQMHistPlotter>:" << std::endl;

  toFile_ = cfg.getParameter<bool>("PrintToFile");
  cfgError_ = 0;

  //--- configure processes
  //std::cout << "--> configuring processes..." << std::endl;
  edm::ParameterSet cfgParSet_processes = cfg.getParameter<edm::ParameterSet>("processes");
  readCfgParameter<cfgEntryProcess>(cfgParSet_processes, processes_);

  //--- check that process types are defined
  //std::cout << "--> checking configuration parameters..." << std::endl;

  int numProcesses_Data = 0;
  int numProcesses_sumMC = 0;
  for (std::map<std::string, cfgEntryProcess>::const_iterator process = processes_.begin(); process != processes_.end();
       ++process) {
    const std::string& type = process->second.type_;

    if (!((type == type_smMC) || (type == type_bsmMC) || (type == type_smSumMC) || (type == type_Data))) {
      edm::LogError("TauDQMHistPlotter") << " Undefined process type = " << type << " !!";
      cfgError_ = 1;
    }

    if (type == type_smSumMC)
      ++numProcesses_sumMC;
    if (type == type_Data)
      ++numProcesses_Data;
  }

  if ((numProcesses_Data > 1) || (numProcesses_sumMC > 1)) {
    edm::LogError("TauDQMHistPlotter") << " Cannot have more than one process of types sumMC and Data !!";
    cfgError_ = 1;
  }

  //--- configure x-axes
  //std::cout << "--> configuring x-axes..." << std::endl;
  edm::ParameterSet cfgParSet_xAxes = cfg.getParameter<edm::ParameterSet>("xAxes");
  readCfgParameter<cfgEntryAxisX>(cfgParSet_xAxes, xAxes_);

  //--- configure y-axes
  //std::cout << "--> configuring y-axes..." << std::endl;
  edm::ParameterSet cfgParSet_yAxes = cfg.getParameter<edm::ParameterSet>("yAxes");
  readCfgParameter<cfgEntryAxisY>(cfgParSet_yAxes, yAxes_);

  //--- configure legends
  //std::cout << "--> configuring legends..." << std::endl;
  edm::ParameterSet cfgParSet_legends = cfg.getParameter<edm::ParameterSet>("legends");
  readCfgParameter<cfgEntryLegend>(cfgParSet_legends, legends_);

  //--- configure labels
  //std::cout << "--> configuring labels..." << std::endl;
  edm::ParameterSet cfgParSet_labels = cfg.getParameter<edm::ParameterSet>("labels");
  readCfgParameter<cfgEntryLabel>(cfgParSet_labels, labels_);

  //--- configure drawOptions
  //std::cout << "--> configuring drawOptions..." << std::endl;
  if (cfg.exists("drawOptionSets")) {
    edm::ParameterSet drawOptionSets = cfg.getParameter<edm::ParameterSet>("drawOptionSets");
    vstring drawOptionSetNames = drawOptionSets.getParameterNamesForType<edm::ParameterSet>();
    for (vstring::const_iterator drawOptionSetName = drawOptionSetNames.begin();
         drawOptionSetName != drawOptionSetNames.end();
         ++drawOptionSetName) {
      edm::ParameterSet drawOptionSet = drawOptionSets.getParameter<edm::ParameterSet>(*drawOptionSetName);

      vstring drawOptionEntryNames = drawOptionSet.getParameterNamesForType<edm::ParameterSet>();
      for (vstring::const_iterator drawOptionEntryName = drawOptionEntryNames.begin();
           drawOptionEntryName != drawOptionEntryNames.end();
           ++drawOptionEntryName) {
        edm::ParameterSet drawOptionEntry = drawOptionSet.getParameter<edm::ParameterSet>(*drawOptionEntryName);

        std::string drawOptionEntryName_full =
            std::string(*drawOptionSetName).append(drawOptionSeparator).append(*drawOptionEntryName);
        drawOptionEntries_.insert(std::pair<std::string, cfgEntryDrawOption>(
            drawOptionEntryName_full, cfgEntryDrawOption(drawOptionEntryName_full, drawOptionEntry)));
      }
    }
  }

  if (cfg.exists("drawOptionEntries")) {
    edm::ParameterSet cfgParSet_drawOptionEntries = cfg.getParameter<edm::ParameterSet>("drawOptionEntries");
    readCfgParameter<cfgEntryDrawOption>(cfgParSet_drawOptionEntries, drawOptionEntries_);
  }

  //--- configure drawJobs
  //std::cout << "--> configuring drawJobs..." << std::endl;
  edm::ParameterSet drawJobs = cfg.getParameter<edm::ParameterSet>("drawJobs");
  vstring drawJobNames = drawJobs.getParameterNamesForType<edm::ParameterSet>();
  for (vstring::const_iterator drawJobName = drawJobNames.begin(); drawJobName != drawJobNames.end(); ++drawJobName) {
    edm::ParameterSet drawJob = drawJobs.getParameter<edm::ParameterSet>(*drawJobName);

    std::map<int, plotDefList> plotDefMap;

    if (drawJob.existsAs<edm::ParameterSet>("plots")) {  // display same monitor element for different processes
      edm::ParameterSet plots = drawJob.getParameter<edm::ParameterSet>("plots");

      vstring dqmMonitorElements = plots.getParameter<vstring>("dqmMonitorElements");
      vstring processes = plots.getParameter<vstring>("processes");

      std::string drawOptionSet = drawJob.getParameter<std::string>("drawOptionSet");
      //std::cout << "drawOptionSet = " << drawOptionSet << std::endl;

      vstring stack = (cfg.exists("stack")) ? drawJob.getParameter<vstring>("stack") : vstring();

      for (vstring::const_iterator process = processes.begin(); process != processes.end(); ++process) {
        int index = 0;
        for (vstring::const_iterator dqmMonitorElement = dqmMonitorElements.begin();
             dqmMonitorElement != dqmMonitorElements.end();
             ++dqmMonitorElement) {
          bool stack_dqmMonitorElement = find_vstring(stack, *process);
          std::string drawOptionEntry = std::string(drawOptionSet).append(drawOptionSeparator).append(*process);
          plotDefMap[index].push_back(
              plotDefEntry(*dqmMonitorElement, drawOptionEntry, "", "", *process, stack_dqmMonitorElement));
          ++index;
        }
      }
    } else {  // display different monitor elements for same process
      typedef std::vector<edm::ParameterSet> vParameterSet;
      vParameterSet plots = drawJob.getParameter<vParameterSet>("plots");

      std::string process = (drawJob.exists("process")) ? drawJob.getParameter<std::string>("process") : "";
      //std::cout << "process (globally set) = " << process << std::endl;

      for (vParameterSet::const_iterator plot = plots.begin(); plot != plots.end(); ++plot) {
        if (process.empty() || plot->exists("process")) {
          process = plot->getParameter<std::string>("process");
          //std::cout << "process (locally set) = " << process << std::endl;
        }

        std::string drawOptionEntry = plot->getParameter<std::string>("drawOptionEntry");
        //std::cout << "drawOptionEntry = " << drawOptionEntry << std::endl;

        std::string legendEntry = "", legendEntryErrorBand = "";
        if (plot->exists("legendEntry")) {
          legendEntry = plot->getParameter<std::string>("legendEntry");
          legendEntryErrorBand = (plot->exists("legendEntryErrorBand"))
                                     ? plot->getParameter<std::string>("legendEntryErrorBand")
                                     : std::string(legendEntry).append(" Uncertainty");
        }
        //std::cout << "legendEntry = " << legendEntry << std::endl;
        //std::cout << "legendEntryErrorBand = " << legendEntryErrorBand << std::endl;

        vstring dqmMonitorElements = plot->getParameter<vstring>("dqmMonitorElements");
        int index = 0;
        for (vstring::const_iterator dqmMonitorElement = dqmMonitorElements.begin();
             dqmMonitorElement != dqmMonitorElements.end();
             ++dqmMonitorElement) {
          plotDefMap[index].push_back(
              plotDefEntry(*dqmMonitorElement, drawOptionEntry, legendEntry, legendEntryErrorBand, process, false));
          ++index;
        }
      }
    }

    //--- check that number of displayed monitor elements is the same for each plot
    unsigned numMonitorElements_ref = 0;
    bool isFirstEntry = true;
    for (std::map<int, plotDefList>::const_iterator plot = plotDefMap.begin(); plot != plotDefMap.end(); ++plot) {
      if (isFirstEntry) {
        numMonitorElements_ref = plot->second.size();
        isFirstEntry = false;
      } else {
        if (plot->second.size() != numMonitorElements_ref) {
          edm::LogError("TauDQMHistPlotter::TauDQMHistPlotter")
              << " Numbers of dqmMonitorElements must be the same for all plots"
              << " --> skipping drawJob = " << (*drawJobName) << " !!";
          cfgError_ = 1;
        }
      }
    }

    //--- expand process directories in names of dqmMonitorElements
    for (std::map<int, plotDefList>::iterator plot = plotDefMap.begin(); plot != plotDefMap.end(); ++plot) {
      for (plotDefList::iterator entry = plot->second.begin(); entry != plot->second.end(); ++entry) {
        std::string dqmMonitorElement = entry->dqmMonitorElement_;
        std::string process = entry->process_;

        std::map<std::string, cfgEntryProcess>::const_iterator it = processes_.find(process);
        if (it != processes_.end()) {
          std::string process_dqmDirectory = it->second.dqmDirectory_;

          //std::cout << "replacing processDir = " << process_dqmDirectory << " in drawJob = " << (*drawJobName) << std::endl;

          int errorFlag = 0;
          std::string dqmMonitorElement_expanded =
              replace_string(dqmMonitorElement, processDirKeyword, process_dqmDirectory, 0, 1, errorFlag);
          //std::cout << " dqmMonitorElement_expanded = " << dqmMonitorElement_expanded << std::endl;

          if (!errorFlag) {
            entry->dqmMonitorElement_ = dqmMonitorElement_expanded;
          } else {
            cfgError_ = 1;
          }
        } else {
          edm::LogError("TauDQMHistPlotter::TauDQMHistPlotter") << " Undefined process = " << process << " !!";
          cfgError_ = 1;
        }
      }
    }

    std::string title = (drawJob.exists("title")) ? drawJob.getParameter<std::string>("title") : "";

    std::string xAxis = drawJob.getParameter<std::string>("xAxis");
    std::string yAxis = drawJob.getParameter<std::string>("yAxis");

    std::string legend = drawJob.getParameter<std::string>("legend");

    vstring labels = (drawJob.exists("labels")) ? drawJob.getParameter<vstring>("labels") : vstring();

    //--- expand parameters in names of dqmMonitorElements;
    //    create drawJob objects
    for (std::map<int, plotDefList>::iterator plot = plotDefMap.begin(); plot != plotDefMap.end(); ++plot) {
      if (drawJob.exists("parameter")) {
        vstring vparameter = drawJob.getParameter<vstring>("parameter");
        //std::cout << "replacing parameter = " << format_vstring(vparameter) << " in drawJob = " << (*drawJobName) << std::endl;

        for (vstring::const_iterator parameter = vparameter.begin(); parameter != vparameter.end(); ++parameter) {
          plotDefList plot_expanded;

          for (plotDefList::const_iterator entry = plot->second.begin(); entry != plot->second.end(); ++entry) {
            std::string dqmMonitorElement = entry->dqmMonitorElement_;

            int errorFlag = 0;
            std::string dqmMonitorElement_expanded =
                replace_string(dqmMonitorElement, parKeyword, *parameter, 1, 1, errorFlag);
            //std::cout << " dqmMonitorElement_expanded = " << dqmMonitorElement_expanded << std::endl;
            if (!errorFlag) {
              plot_expanded.push_back(plotDefEntry(dqmMonitorElement_expanded,
                                                   entry->drawOptionEntry_,
                                                   entry->legendEntry_,
                                                   entry->legendEntryErrorBand_,
                                                   entry->process_,
                                                   entry->doStack_));
            } else {
              cfgError_ = 1;
            }
          }

          int errorFlag = 0;
          std::string title_expanded = replace_string(title, parKeyword, *parameter, 0, 1, errorFlag);
          //std::cout << " title_expanded = " << title_expanded << std::endl;
          std::string xAxis_expanded = replace_string(xAxis, parKeyword, *parameter, 0, 1, errorFlag);
          //std::cout << " xAxis_expanded = " << xAxis_expanded << std::endl;
          std::string yAxis_expanded = replace_string(yAxis, parKeyword, *parameter, 0, 1, errorFlag);
          //std::cout << " yAxis_expanded = " << yAxis_expanded << std::endl;
          if (errorFlag)
            cfgError_ = 1;

          drawJobs_.push_back(cfgEntryDrawJob(std::string(*drawJobName).append(*parameter),
                                              plot_expanded,
                                              title_expanded,
                                              xAxis_expanded,
                                              yAxis_expanded,
                                              legend,
                                              labels));
        }
      } else {
        drawJobs_.push_back(cfgEntryDrawJob(*drawJobName, plot->second, title, xAxis, yAxis, legend, labels));
      }
    }
  }

  //--- check that all information neccessary to process drawJob is defined;
  for (std::list<cfgEntryDrawJob>::const_iterator drawJob = drawJobs_.begin(); drawJob != drawJobs_.end(); ++drawJob) {
    for (plotDefList::const_iterator plot = drawJob->plots_.begin(); plot != drawJob->plots_.end(); ++plot) {
      checkCfgDef<cfgEntryDrawOption>(
          plot->drawOptionEntry_, drawOptionEntries_, cfgError_, "drawOptionEntry", drawJob->name_);
      checkCfgDef<cfgEntryProcess>(plot->process_, processes_, cfgError_, "process", drawJob->name_);
    }

    checkCfgDef<cfgEntryAxisX>(drawJob->xAxis_, xAxes_, cfgError_, "xAxis", drawJob->name_);
    checkCfgDef<cfgEntryAxisY>(drawJob->yAxis_, yAxes_, cfgError_, "yAxis", drawJob->name_);

    checkCfgDef<cfgEntryLegend>(drawJob->legend_, legends_, cfgError_, "legend", drawJob->name_);

    checkCfgDefs<cfgEntryLabel>(drawJob->labels_, labels_, cfgError_, "label", drawJob->name_);
  }

  //--- configure canvas size
  //std::cout << "--> configuring canvas size..." << std::endl;
  canvasSizeX_ = (cfg.exists("canvasSizeX")) ? cfg.getParameter<int>("canvasSizeX") : defaultCanvasSizeX;
  canvasSizeY_ = (cfg.exists("canvasSizeY")) ? cfg.getParameter<int>("canvasSizeY") : defaultCanvasSizeY;

  //--- configure output files
  //std::cout << "--> configuring postscript output file..." << std::endl;

  outputFilePath_ = (cfg.exists("outputFilePath")) ? cfg.getParameter<std::string>("outputFilePath") : "";
  if (outputFilePath_.rbegin() != outputFilePath_.rend()) {
    if ((*outputFilePath_.rbegin()) == '/')
      outputFilePath_.erase(outputFilePath_.length() - 1);
  }
  //std::cout << " outputFilePath = " << outputFilePath_ << std::endl;

  outputFileName_ = (cfg.exists("outputFileName")) ? cfg.getParameter<std::string>("outputFileName") : "";
  //std::cout << " outputFileName = " << outputFileName_ << std::endl;

  indOutputFileName_ = (cfg.exists("indOutputFileName")) ? cfg.getParameter<std::string>("indOutputFileName") : "";
  if (!indOutputFileName_.empty() && indOutputFileName_.find('.') == std::string::npos) {
    edm::LogError("TauDQMHistPlotter") << " Failed to determine type of graphics format from indOutputFileName = "
                                       << indOutputFileName_ << " !!";
    cfgError_ = 1;
  }
  //std::cout << " indOutputFileName = " << indOutputFileName_ << std::endl;

  //--- check that exactly one type of output is specified for the plots
  //    (either separate graphics files displaying one plot each
  //     or postscript file displaying all plots on successive pages;
  //     cannot create both types of output simultaneously,
  //     as TCanvas::Print seems to interfere with TPostScript::NewPage)
  if (outputFileName_.empty() && indOutputFileName_.empty()) {
    edm::LogError("TauDQMHistPlotter") << " Either outputFileName or indOutputFileName must be specified !!";
    cfgError_ = 1;
  }

  if (!outputFileName_.empty() && !indOutputFileName_.empty()) {
    edm::LogError("TauDQMHistPlotter") << " Must not specify outputFileName and indOutputFileName simultaneously !!";
    cfgError_ = 1;
  }

  if (verbosity)
    std::cout << "done." << std::endl;
}

TauDQMHistPlotter::~TauDQMHistPlotter() {
  // nothing to be done yet...
}

void TauDQMHistPlotter::analyze(const edm::Event&, const edm::EventSetup&) {
  // nothing to be done yet...
}

void TauDQMHistPlotter::endRun(const edm::Run& r, const edm::EventSetup& c) {
  if (verbosity)
    std::cout << "<TauDQMHistPlotter::endJob>:" << std::endl;

  //--- check that configuration parameters contain no errors
  if (cfgError_) {
    edm::LogError("endJob") << " Error in Configuration ParameterSet --> histograms will NOT be plotted !!";
    return;
  }

  //--- check that DQMStore service is available
  if (!edm::Service<DQMStore>().isAvailable()) {
    edm::LogError("endJob") << " Failed to access dqmStore --> histograms will NOT be plotted !!";
    return;
  }

  DQMStore& dqmStore = (*edm::Service<DQMStore>());

  //--- stop ROOT from keeping references to all hsitograms
  //TH1::AddDirectory(false);

  //--- stop ROOT from opening X-window for canvas output
  //    (in order to be able to run in batch mode)
  gROOT->SetBatch(true);

  //--- initialize graphical output;
  //    open postscript file
  TCanvas canvas("TauDQMHistPlotter", "TauDQMHistPlotter", canvasSizeX_, canvasSizeY_);
  canvas.SetFillColor(10);

  //--- restrict area in which histograms are drawn to quadratic TPad in the center of the TCanvas,
  //    in order to make space for axis labels...
  //TPad pad("EWKTauPad", "EWKTauPad", 0.02, 0.15, 0.98, 0.85);
  //pad.SetFillColor(10);
  //pad.Draw();
  //pad.Divide(1,1);
  //pad.cd(1);

  TPostScript* ps = nullptr;
  if (!outputFileName_.empty()) {
    std::string psFileName =
        (!outputFilePath_.empty()) ? std::string(outputFilePath_).append("/").append(outputFileName_) : outputFileName_;
    ps = new TPostScript(psFileName.data(), 112);
  }

  //--- process drawJobs
  for (std::list<cfgEntryDrawJob>::const_iterator drawJob = drawJobs_.begin(); drawJob != drawJobs_.end(); ++drawJob) {
    const std::string& drawJobName = drawJob->name_;
    if (verbosity)
      std::cout << "--> processing drawJob " << drawJobName << "..." << std::endl;

    //--- prepare internally used histogram data-structures
    TH1* stackedHistogram_sum = nullptr;
    std::list<TH1*> histogramsToDelete;
    std::list<plotDefEntry*> drawOptionsToDelete;

    typedef std::pair<TH1*, const plotDefEntry*> histogram_drawOption_pair;
    std::list<histogram_drawOption_pair> allHistograms;

    for (plotDefList::const_iterator plot = drawJob->plots_.begin(); plot != drawJob->plots_.end(); ++plot) {
      std::string dqmMonitorElementName_full =
          dqmDirectoryName(std::string(dqmRootDirectory)).append(plot->dqmMonitorElement_);
      if (verbosity)
        std::cout << " dqmMonitorElementName_full = " << dqmMonitorElementName_full << std::endl;
      MonitorElement* dqmMonitorElement = dqmStore.get(dqmMonitorElementName_full);

      TH1* histogram = dqmMonitorElement->getTH1F();
      if (verbosity)
        std::cout << "Got Histogram " << std::endl;
      //      TH1* histogram = ( dqmMonitorElement ) ? dynamic_cast<TH1*>(dqmMonitorElement->getTH1()->Clone()) : NULL;
      //histogramsToDelete.push_back(histogram);

      if (histogram == nullptr) {
        edm::LogError("endJob") << " Failed to access dqmMonitorElement = " << dqmMonitorElementName_full << ","
                                << " needed by drawJob = " << drawJobName << " --> histograms will NOT be plotted !!";
        continue;
      }

      if (!histogram->GetSumw2N())
        histogram->Sumw2();

      const cfgEntryDrawOption* drawOptionConfig =
          findCfgDef<cfgEntryDrawOption>(plot->drawOptionEntry_, drawOptionEntries_, "drawOptionEntry", drawJobName);
      if (drawOptionConfig == nullptr) {
        edm::LogError("endJob") << " Failed to access information needed by drawJob = " << drawJobName
                                << " --> histograms will NOT be plotted !!";
        return;
      }

      if (drawOptionConfig->drawOption_ == drawOption_eBand) {
        //--- add histogram displaying central value as solid line
        TH1* histogram_centralValue = dynamic_cast<TH1*>(histogram->Clone());
        histogram_centralValue->SetName(std::string(histogram->GetName()).append("_centralValue").data());
        cfgEntryDrawOption drawOptionConfig_centralValue(*drawOptionConfig);
        drawOptionConfig_centralValue.fillColor_ = 0;
        drawOptionConfig_centralValue.fillStyle_ = 0;
        drawOptionConfig_centralValue.drawOption_ = "hist";
        drawOptionConfig_centralValue.drawOptionLegend_ = "l";
        std::string drawOptionName_centralValue = std::string(plot->drawOptionEntry_).append("_centralValue");
        //--- entries in std::map need to be unique,
        //    so need to check whether drawOptionEntry already exists...
        if (drawOptionEntries_.find(drawOptionName_centralValue) == drawOptionEntries_.end())
          drawOptionEntries_.insert(std::pair<std::string, cfgEntryDrawOption>(
              drawOptionName_centralValue,
              cfgEntryDrawOption(drawOptionName_centralValue, drawOptionConfig_centralValue)));
        plotDefEntry* plot_centralValue = new plotDefEntry(*plot);
        plot_centralValue->drawOptionEntry_ = drawOptionName_centralValue;
        allHistograms.push_back(histogram_drawOption_pair(histogram_centralValue, plot_centralValue));
        histogramsToDelete.push_back(histogram_centralValue);
        drawOptionsToDelete.push_back(plot_centralValue);

        //--- add histogram displaying uncertainty as shaded error band
        TH1* histogram_ErrorBand = dynamic_cast<TH1*>(histogram->Clone());
        histogram_ErrorBand->SetName(std::string(histogram->GetName()).append("_ErrorBand").data());
        cfgEntryDrawOption drawOptionConfig_ErrorBand(*drawOptionConfig);
        drawOptionConfig_ErrorBand.markerColor_ = drawOptionConfig_ErrorBand.fillColor_;
        drawOptionConfig_ErrorBand.markerSize_ = 0.;
        drawOptionConfig_ErrorBand.lineColor_ = drawOptionConfig_ErrorBand.fillColor_;
        drawOptionConfig_ErrorBand.lineWidth_ = 0;
        drawOptionConfig_ErrorBand.drawOption_ = "e2";
        drawOptionConfig_ErrorBand.drawOptionLegend_ = "f";
        std::string drawOptionName_ErrorBand = std::string(plot->drawOptionEntry_).append("_ErrorBand");
        //--- entries in std::map need to be unique,
        //    so need to check whether drawOptionEntry already exists...
        if (drawOptionEntries_.find(drawOptionName_ErrorBand) == drawOptionEntries_.end())
          drawOptionEntries_.insert(std::pair<std::string, cfgEntryDrawOption>(
              drawOptionName_ErrorBand, cfgEntryDrawOption(drawOptionName_ErrorBand, drawOptionConfig_ErrorBand)));
        plotDefEntry* plot_ErrorBand = new plotDefEntry(*plot);
        plot_ErrorBand->drawOptionEntry_ = drawOptionName_ErrorBand;
        plot_ErrorBand->isErrorBand_ = true;
        allHistograms.push_back(histogram_drawOption_pair(histogram_ErrorBand, plot_ErrorBand));
        histogramsToDelete.push_back(histogram_ErrorBand);
        drawOptionsToDelete.push_back(plot_ErrorBand);
      } else if (plot->doStack_) {
        TH1* stackedHistogram = dynamic_cast<TH1*>(histogram->Clone());
        if (stackedHistogram_sum)
          stackedHistogram->Add(stackedHistogram_sum);
        stackedHistogram_sum = stackedHistogram;
        histogramsToDelete.push_back(stackedHistogram);
        allHistograms.push_back(histogram_drawOption_pair(stackedHistogram, &(*plot)));
      } else {
        allHistograms.push_back(histogram_drawOption_pair(histogram, &(*plot)));
      }
    }

    //--- determine normalization of y-axis
    //    (maximum of any of the histograms included in drawJob)
    double yAxisNorm = 0.;
    for (std::list<histogram_drawOption_pair>::const_iterator it = allHistograms.begin(); it != allHistograms.end();
         ++it) {
      yAxisNorm = TMath::Max(yAxisNorm, it->first->GetMaximum());
    }
    //std::cout << " yAxisNorm = " << yAxisNorm << std::endl;

    //--- prepare histograms for drawing
    const cfgEntryAxisX* xAxisConfig = findCfgDef<cfgEntryAxisX>(drawJob->xAxis_, xAxes_, "xAxis", drawJobName);
    const cfgEntryAxisY* yAxisConfig = findCfgDef<cfgEntryAxisY>(drawJob->yAxis_, yAxes_, "yAxis", drawJobName);
    const cfgEntryLegend* legendConfig = findCfgDef<cfgEntryLegend>(drawJob->legend_, legends_, "legend", drawJobName);
    if (xAxisConfig == nullptr || yAxisConfig == nullptr || legendConfig == nullptr) {
      edm::LogError("endJob") << " Failed to access information needed by drawJob = " << drawJobName
                              << " --> histograms will NOT be plotted !!";
      return;
    }

    //--- WARNING: need to call
    //              TLegend::TLegend(Double_t, Double_t,Double_t, Double_t, const char* = "", Option_t* = "brNDC")
    //             constructor, as TLegend::TLegend default constructor causes the created TLegend object to behave differently !!
    TLegend legend(defaultLegendPosX,
                   defaultLegendPosY,
                   defaultLegendPosX + defaultLegendSizeX,
                   defaultLegendPosY + defaultLegendSizeY);
    legendConfig->applyTo(&legend);

    std::list<histoDrawEntry> smProcessHistogramList;
    std::list<histoDrawEntry> bsmProcessHistogramList;
    std::list<histoDrawEntry> smSumHistogramList;
    std::list<histoDrawEntry> smSumUncertaintyHistogramList;
    std::list<histoDrawEntry> dataHistogramList;

    for (std::list<histogram_drawOption_pair>::const_iterator it = allHistograms.begin(); it != allHistograms.end();
         ++it) {
      TH1* histogram = it->first;
      const plotDefEntry* drawOption = it->second;

      const cfgEntryDrawOption* drawOptionConfig = findCfgDef<cfgEntryDrawOption>(
          drawOption->drawOptionEntry_, drawOptionEntries_, "drawOptionEntry", drawJobName);
      const cfgEntryProcess* processConfig =
          findCfgDef<cfgEntryProcess>(drawOption->process_, processes_, "process", drawJobName);
      if (drawOptionConfig == nullptr || processConfig == nullptr) {
        edm::LogError("endJob") << " Failed to access information needed by drawJob = " << drawJobName
                                << " --> histograms will NOT be plotted !!";
        return;
      }

      if (!drawJob->title_.empty())
        histogram->SetTitle(drawJob->title_.data());

      xAxisConfig->applyTo(histogram);
      yAxisConfig->applyTo(histogram, yAxisNorm);

      bool yLogScale = (yAxisConfig->yScale_ == yScale_log) ? true : false;
      //std::cout << " yLogScale = " << yLogScale << std::endl;
      //pad.SetLogy(yLogScale);
      canvas.SetLogy(yLogScale);

      drawOptionConfig->applyTo(histogram);
      histogram->SetStats(false);

      if (drawOption->isErrorBand_) {
        smSumUncertaintyHistogramList.push_back(histoDrawEntry(histogram, drawOptionConfig->drawOption_.data()));
      } else {
        if (processConfig->type_ == type_smMC) {
          smProcessHistogramList.push_back(histoDrawEntry(histogram, drawOptionConfig->drawOption_.data()));
        } else if (processConfig->type_ == type_bsmMC) {
          bsmProcessHistogramList.push_back(histoDrawEntry(histogram, drawOptionConfig->drawOption_.data()));
        } else if (processConfig->type_ == type_smSumMC) {
          smSumHistogramList.push_back(histoDrawEntry(histogram, drawOptionConfig->drawOption_.data()));
        } else if (processConfig->type_ == type_Data) {
          dataHistogramList.push_back(histoDrawEntry(histogram, drawOptionConfig->drawOption_.data()));
        }
      }

      std::string legendEntry, legendDrawOption;
      if (drawOption->isErrorBand_) {
        legendEntry = (!drawOption->legendEntryErrorBand_.empty()) ? drawOption->legendEntryErrorBand_
                                                                   : processConfig->legendEntryErrorBand_;
        legendDrawOption = "f";
      } else {
        legendEntry = (!drawOption->legendEntry_.empty()) ? drawOption->legendEntry_ : processConfig->legendEntry_;
        legendDrawOption = drawOptionConfig->drawOptionLegend_;
      }

      legend.AddEntry(histogram, legendEntry.data(), legendDrawOption.data());
    }

    std::list<TPaveText> labels;
    for (vstring::const_iterator labelName = drawJob->labels_.begin(); labelName != drawJob->labels_.end();
         ++labelName) {
      const cfgEntryLabel* labelConfig = findCfgDef<cfgEntryLabel>(*labelName, labels_, "label", drawJobName);

      TPaveText label;
      labelConfig->applyTo(&label);

      labels.push_back(label);
    }

    //--- draw histograms
    //   - in the order:
    //    1. uncertainty on sum of all Standard Model processes
    //    2. sum of all Standard Model processes
    //    3. individual Standard Model processes
    //    4. individual beyond the Standard Model processes
    //    5. data
    bool isFirstHistogram = true;
    drawHistograms(smSumUncertaintyHistogramList, isFirstHistogram);
    drawHistograms(smSumHistogramList, isFirstHistogram);

    //--- process histograms for individual Standard Model processes
    //    in reverse order, so that most stacked histogram gets drawn first
    for (std::list<histoDrawEntry>::reverse_iterator it = smProcessHistogramList.rbegin();
         it != smProcessHistogramList.rend();
         ++it) {
      std::string drawOption = (isFirstHistogram) ? it->second : std::string(it->second).append("same");
      it->first->Draw(drawOption.data());
      isFirstHistogram = false;
    }

    drawHistograms(bsmProcessHistogramList, isFirstHistogram);
    drawHistograms(dataHistogramList, isFirstHistogram);

    legend.Draw();

    for (std::list<TPaveText>::iterator label = labels.begin(); label != labels.end(); ++label) {
      label->Draw();
    }

    //pad.RedrawAxis();

    canvas.Update();
    //pad.Update();

    if (!indOutputFileName_.empty() && toFile_) {
      int errorFlag = 0;
      std::string modIndOutputFileName = replace_string(indOutputFileName_, plotKeyword, drawJobName, 1, 1, errorFlag);
      if (!errorFlag) {
        std::string fullFileName = (!outputFilePath_.empty())
                                       ? std::string(outputFilePath_).append("/").append(modIndOutputFileName)
                                       : modIndOutputFileName;
        canvas.Print(fullFileName.data());
      } else {
        edm::LogError("endJob") << " Failed to decode indOutputFileName = " << indOutputFileName_ << " --> skipping !!";
      }
    }

    if (ps)
      ps->NewPage();

    //--- delete temporarily created histogram and drawOption objects
    for (std::list<TH1*>::const_iterator histogram = histogramsToDelete.begin(); histogram != histogramsToDelete.end();
         ++histogram) {
      delete (*histogram);
    }

    for (std::list<plotDefEntry*>::const_iterator drawOption = drawOptionsToDelete.begin();
         drawOption != drawOptionsToDelete.end();
         ++drawOption) {
      delete (*drawOption);
    }
  }

  //--- close postscript file
  canvas.Clear();
  if (verbosity)
    std::cout << "done." << std::endl;
  if (ps)
    ps->Close();
  delete ps;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TauDQMHistPlotter);
