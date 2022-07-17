#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingCalo.h"

#include <TH1F.h>

#include <iostream>
#include <string>
#include <vector>

//#define EDM_ML_DEBUG

class HGCalTBMBAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  HGCalTBMBAnalyzer(const edm::ParameterSet &);
  ~HGCalTBMBAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void analyze(edm::Event const &, edm::EventSetup const &) override;

  const std::vector<std::string> listNames_;
  const edm::InputTag labelMBCalo_;
  const unsigned int nList_;
  const edm::EDGetTokenT<MaterialAccountingCaloCollection> tokMBCalo_;
  std::vector<TH1D *> me100_, me200_, me300_;
};

HGCalTBMBAnalyzer::HGCalTBMBAnalyzer(const edm::ParameterSet &p)
    : listNames_(p.getParameter<std::vector<std::string>>("detectorNames")),
      labelMBCalo_(p.getParameter<edm::InputTag>("labelMBCalo")),
      nList_(listNames_.size()),
      tokMBCalo_(consumes<MaterialAccountingCaloCollection>(labelMBCalo_)) {
  edm::LogVerbatim("HGCSim") << "HGCalTBMBAnalyzer initialized for " << nList_ << " volumes";
  for (unsigned int k = 0; k < nList_; ++k)
    edm::LogVerbatim("HGCSim") << " [" << k << "] " << listNames_[k];

  edm::Service<TFileService> tfile;
  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  char name[20], title[80];
  TH1D *hist;
  for (unsigned int i = 0; i <= nList_; i++) {
    std::string named = (i == nList_) ? "Total" : listNames_[i];
    sprintf(name, "RadL%d", i);
    sprintf(title, "MB(X0) for (%s)", named.c_str());
    hist = tfile->make<TH1D>(name, title, 100000, 0.0, 100.0);
    hist->Sumw2(true);
    me100_.push_back(hist);
    sprintf(name, "IntL%d", i);
    sprintf(title, "MB(L0) for (%s)", named.c_str());
    hist = tfile->make<TH1D>(name, title, 100000, 0.0, 10.0);
    hist->Sumw2(true);
    me200_.push_back(hist);
    sprintf(name, "StepL%d", i);
    sprintf(title, "MB(Step) for (%s)", named.c_str());
    hist = tfile->make<TH1D>(name, title, 100000, 0.0, 50000.0);
    hist->Sumw2(true);
    me300_.push_back(hist);
  }
  edm::LogVerbatim("HGCSim") << "HGCalTBMBAnalyzer: Booking user histos done ===";
}

void HGCalTBMBAnalyzer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> names = {"HGCalBeamWChamb",
                                    "HGCalBeamS1",
                                    "HGCalBeamS2",
                                    "HGCalBeamS3",
                                    "HGCalBeamS4",
                                    "HGCalBeamS5",
                                    "HGCalBeamS6",
                                    "HGCalBeamCK3",
                                    "HGCalBeamHaloCounter",
                                    "HGCalBeamMuonCounter",
                                    "HGCalEE",
                                    "HGCalHE",
                                    "HGCalAH"};
  desc.add<std::vector<std::string>>("detectorNames", names);
  desc.add<edm::InputTag>("labelMBCalo", edm::InputTag("g4SimHits", "HGCalTBMB"));
  descriptions.add("hgcalTBMBAnalyzer", desc);
}

void HGCalTBMBAnalyzer::analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Run " << iEvent.id().run() << " Event " << iEvent.id().event() << " Luminosity "
                             << iEvent.luminosityBlock() << " Bunch " << iEvent.bunchCrossing();
#endif

  // Fill from the MB collection
  auto const &hgcalMBColl = iEvent.getHandle(tokMBCalo_);
  if (hgcalMBColl.isValid()) {
    auto hgcalMB = hgcalMBColl.product();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "Finds MaterialBudegetCollection with " << hgcalMB->size() << " entries";
#endif

    for (auto itr = hgcalMB->begin(); itr != hgcalMB->end(); ++itr) {
      for (uint32_t ii = 0; ii < itr->m_stepLen.size(); ++ii) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "HGCalTBMBAnalyzer:index " << ii << " integrated  step " << itr->m_stepLen[ii]
                                   << " X0 " << itr->m_radLen[ii] << " Lamda " << itr->m_intLen[ii];
#endif
        if (ii < nList_) {
          me100_[ii]->Fill(itr->m_radLen[ii]);
          me200_[ii]->Fill(itr->m_intLen[ii]);
          me300_[ii]->Fill(itr->m_stepLen[ii]);
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(HGCalTBMBAnalyzer);
