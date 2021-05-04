// -*- C++ -*-
//
// Package:    Validation/RecoParticleFlow
// Class:      OffsetDQMPostProcessor.cc
//
// Original Author:  "Kenichi Hatakeyama"
//

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

//
// class decleration
//

class OffsetDQMPostProcessor : public DQMEDHarvester {
public:
  explicit OffsetDQMPostProcessor(const edm::ParameterSet&);
  ~OffsetDQMPostProcessor() override;

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

  std::string offsetPlotBaseName;
  std::string offsetDir;
  float offsetR;
  std::vector<std::string> pftypes;
  std::vector<std::string> offsetVariableTypes;
  int muHigh;
  int npvHigh;

  bool debug = false;
};

//
// constructors and destructor
//
OffsetDQMPostProcessor::OffsetDQMPostProcessor(const edm::ParameterSet& iConfig) {
  offsetPlotBaseName = iConfig.getParameter<std::string>("offsetPlotBaseName");
  offsetDir = iConfig.getParameter<std::string>("offsetDir");
  offsetVariableTypes = iConfig.getParameter<std::vector<std::string> >("offsetVariableTypes");
  offsetR = iConfig.getUntrackedParameter<double>("offsetR");
  pftypes = iConfig.getParameter<std::vector<std::string> >("pftypes");
  muHigh = iConfig.getUntrackedParameter<int>("muHigh");
  npvHigh = iConfig.getUntrackedParameter<int>("npvHigh");
};

OffsetDQMPostProcessor::~OffsetDQMPostProcessor() {}

// ------------ method called right after a run ends ------------
void OffsetDQMPostProcessor::dqmEndJob(DQMStore::IBooker& ibook_, DQMStore::IGetter& iget_) {
  iget_.setCurrentFolder(offsetDir);

  std::string stitle;
  std::vector<MonitorElement*> vME;
  std::vector<std::string> MEStrings = iget_.getMEs();
  std::for_each(MEStrings.begin(), MEStrings.end(), [&](auto& s) { s.insert(0, offsetDir); });

  // temporary ME and root objects
  MonitorElement* mtmp;
  TProfile* hproftmp;
  TH1F* htmp;
  TH1F* hscaled;

  //
  // Offset plots vs eta
  //
  for (std::vector<std::string>::const_iterator i = offsetVariableTypes.begin(); i != offsetVariableTypes.end(); ++i) {
    //
    // getting the average value for Npv and mu
    //
    stitle = offsetDir + (*i);
    std::vector<std::string>::const_iterator it = std::find(MEStrings.begin(), MEStrings.end(), stitle);
    if (it == MEStrings.end())
      continue;
    mtmp = iget_.get(stitle);
    float avg = mtmp->getMean();
    int iavg = int(avg + 0.5);  // integer version for identifying correcping ME, in order to get the rounding correctly

    if (avg < 1.)
      avg = 1.;  // protection against this value going too low

    if (iavg < 0)
      iavg = 0;  // checking lower bound (avoid division by zero)
    if (*i == "npv" && iavg >= npvHigh)
      iavg = npvHigh - 1;  // checking upper bound
    else if (*i == "mu" && iavg >= muHigh)
      iavg = muHigh - 1;  //

    //
    // storing the value
    //
    stitle = (*i) + "_mean";
    MonitorElement* MEmean = ibook_.bookFloat(stitle);
    MEmean->Fill(avg);
    vME.push_back(MEmean);

    //
    // for each pf types
    //
    for (std::vector<std::string>::const_iterator j = pftypes.begin(); j != pftypes.end(); ++j) {
      // accessing profiles
      std::string str_base = *i + std::to_string(iavg);
      if ((*i) == "npv")
        stitle = offsetDir + "npvPlots/" + str_base + "/" + offsetPlotBaseName + "_" + str_base + "_" + (*j);
      else if ((*i) == "mu")
        stitle = offsetDir + "muPlots/" + str_base + "/" + offsetPlotBaseName + "_" + str_base + "_" + (*j);
      else
        return;

      // making scaled plot and ME
      mtmp = iget_.get(stitle);
      hproftmp = (TProfile*)mtmp->getTProfile();
      htmp = (TH1F*)hproftmp->ProjectionX();
      TAxis* xaxis = (TAxis*)htmp->GetXaxis();
      stitle = offsetPlotBaseName + "_" + str_base + "_" + *j;
      hscaled = new TH1F(stitle.c_str(), stitle.c_str(), xaxis->GetNbins(), xaxis->GetXbins()->GetArray());

      htmp->Scale(pow(offsetR, 2) / 2. / float(avg));             // pi*R^2 / (deltaEta*2pi) / <mu or NPV>
      for (int ibin = 1; ibin <= hscaled->GetNbinsX(); ibin++) {  // 1/deltaEta part
        hscaled->SetBinContent(ibin, htmp->GetBinContent(ibin) / htmp->GetBinWidth(ibin));
        hscaled->SetBinError(ibin, htmp->GetBinError(ibin) / htmp->GetBinWidth(ibin));
      }

      // storing new ME
      stitle = offsetPlotBaseName + "_" + *i + "_" + *j;
      mtmp = ibook_.book1D(stitle.c_str(), hscaled);
      vME.push_back(mtmp);
    }
  }

  //
  // Checks
  //
  if (debug) {
    for (std::vector<MonitorElement*>::const_iterator i = vME.begin(); i != vME.end(); ++i)
      (*i)->getTH1F()->Print();
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OffsetDQMPostProcessor);
