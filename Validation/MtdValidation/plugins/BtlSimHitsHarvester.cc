// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlSimHitsHarvester
//
/**\class BtlSimHitsHarvester BtlSimHitsHarvester.cc Validation/MtdValidation/plugins/BtlSimHitsHarvester.cc

 Description: BTL SIM hits validation harvester

 Implementation:
     [Notes on implementation]
*/

#include <string>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

class BtlSimHitsHarvester : public DQMEDHarvester {
public:
  explicit BtlSimHitsHarvester(const edm::ParameterSet& iConfig);
  ~BtlSimHitsHarvester() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  const std::string folder_;

  // --- Histograms
  MonitorElement* meHitOccupancy_;
  MonitorElement* meHitOccupancyCell_;
  MonitorElement* meHitOccupancySM_;
  static constexpr int nRU_ = 6;
  MonitorElement* meHitOccupancyRUSlice_[nRU_];
  MonitorElement* meHitOccupancyCellRUSlice_[nRU_];
  MonitorElement* meHitOccupancySMRUSlice_[nRU_];
};

// ------------ constructor and destructor --------------
BtlSimHitsHarvester::BtlSimHitsHarvester(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {}

BtlSimHitsHarvester::~BtlSimHitsHarvester() {}

// ------------ endjob tasks ----------------------------
void BtlSimHitsHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& igetter) {
  // --- Get the monitoring histograms
  MonitorElement* meBtlHitLogEnergy = igetter.get(folder_ + "BtlHitLogEnergy");
  MonitorElement* meBtlHitLogEnergyRUSlice[nRU_];
  MonitorElement* meNevents = igetter.get(folder_ + "BtlNevents");
  MonitorElement* meBtlHitMultCell = igetter.get(folder_ + "BtlHitMultCell");
  MonitorElement* meBtlHitMultCellRUSlice[nRU_];
  MonitorElement* meBtlHitMultSM = igetter.get(folder_ + "BtlHitMultSM");
  MonitorElement* meBtlHitMultSMRUSlice[nRU_];
  bool missing_ru_slice = false;
  for (unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    meBtlHitLogEnergyRUSlice[ihistoRU] = igetter.get(folder_ + "BtlHitLogEnergyRUSlice" + std::to_string(ihistoRU));
    meBtlHitMultCellRUSlice[ihistoRU] = igetter.get(folder_ + "BtlHitMultCellRUSlice" + std::to_string(ihistoRU));
    meBtlHitMultSMRUSlice[ihistoRU] = igetter.get(folder_ + "BtlHitMultSMRUSlice" + std::to_string(ihistoRU));
    if (!meBtlHitLogEnergyRUSlice[ihistoRU] || !meBtlHitMultCellRUSlice[ihistoRU] || !meBtlHitMultSMRUSlice[ihistoRU]) {
      missing_ru_slice = true;
    }
  }
  if (!meBtlHitLogEnergy || !meNevents || !meBtlHitMultCell || missing_ru_slice || !meBtlHitMultSM) {
    edm::LogError("BtlSimHitsHarvester") << "Monitoring histograms not found!" << std::endl;
    return;
  }

  // --- Get the number of BTL crystals and the number of processed events
  const float NBtlCrystals = BTLDetId::kCrystalsBTL;
  const float NBtlSMs = NBtlCrystals / BTLDetId::kCrystalsPerModuleV2;
  const float Nevents = meNevents->getEntries();
  const float scale_Crystals = (Nevents > 0 ? 1. / (Nevents * NBtlCrystals) : 1.);
  const float scale_Crystals_RU = (Nevents > 0 ? 1. / (Nevents * NBtlCrystals / nRU_) : 1.);
  const float scale_SMs = (Nevents > 0 ? 1. / (Nevents * NBtlSMs) : 1.);
  const float scale_SMs_RU = (Nevents > 0 ? 1. / (Nevents * NBtlSMs / nRU_) : 1.);

  // --- Book histograms
  ibook.cd(folder_);
  meHitOccupancy_ = ibook.book1D("BtlHitOccupancy",
                                 "BTL cell occupancy vs hit energy;log_{10}(E_{SIM} [MeV]); Occupancy per event",
                                 meBtlHitLogEnergy->getNbinsX(),
                                 meBtlHitLogEnergy->getTH1()->GetXaxis()->GetXmin(),
                                 meBtlHitLogEnergy->getTH1()->GetXaxis()->GetXmax());
  meHitOccupancyCell_ = ibook.book1D("BtlHitOccupancyCell",
                                     "BTL cell occupancy vs energy threshold;log_{10}(E_{th} [MeV]); Occupancy per event",
                                     meBtlHitMultCell->getNbinsX(),
                                     meBtlHitMultCell->getTH1()->GetXaxis()->GetXmin(),
                                     meBtlHitMultCell->getTH1()->GetXaxis()->GetXmax());
  meHitOccupancySM_ = ibook.book1D("BtlHitOccupancySM",
                                   "BTL SM occupancy vs energy threshold;log_{10}(E_{th} [MeV]); Occupancy per event",
                                   meBtlHitMultSM->getNbinsX(),
                                   meBtlHitMultSM->getTH1()->GetXaxis()->GetXmin(),
                                   meBtlHitMultSM->getTH1()->GetXaxis()->GetXmax());
  for(unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    std::string name_LogEnergy = "BtlHitOccupancyRUSlice" + std::to_string(ihistoRU);
    std::string title_LogEnergy = "BTL cell occupancy vs hit energy (RU " + std::to_string(ihistoRU) + ");log_{10}(E_{SIM} [MeV]); Occupancy per event";
    meHitOccupancyRUSlice_[ihistoRU] = ibook.book1D(name_LogEnergy,
                                                    title_LogEnergy,
                                                    meBtlHitLogEnergyRUSlice[ihistoRU]->getNbinsX(),
                                                    meBtlHitLogEnergyRUSlice[ihistoRU]->getTH1()->GetXaxis()->GetXmin(),
                                                    meBtlHitLogEnergyRUSlice[ihistoRU]->getTH1()->GetXaxis()->GetXmax());
    std::string name_cell = "BtlHitOccupancyCellRUSlice" + std::to_string(ihistoRU);
    std::string title_cell = "BTL cell occupancy vs energy threshold (RU " + std::to_string(ihistoRU) + ");log_{10}(E_{th} [MeV]); Occupancy per event";
    meHitOccupancyCellRUSlice_[ihistoRU] = ibook.book1D(name_cell,
                                                    title_cell,
                                                    meBtlHitMultCellRUSlice[ihistoRU]->getNbinsX(),
                                                    meBtlHitMultCellRUSlice[ihistoRU]->getTH1()->GetXaxis()->GetXmin(),
                                                    meBtlHitMultCellRUSlice[ihistoRU]->getTH1()->GetXaxis()->GetXmax());
    std::string name_SM = "BtlHitOccupancySMRUSlice" + std::to_string(ihistoRU);
    std::string title_SM = "BTL SM occupancy vs energy threshold(RU " + std::to_string(ihistoRU) + ");log_{10}(E_{th} [MeV]); Occupancy per event";
    meHitOccupancySMRUSlice_[ihistoRU] = ibook.book1D(name_SM,
                                                      title_SM,
                                                      meBtlHitMultSMRUSlice[ihistoRU]->getNbinsX(),
                                                      meBtlHitMultSMRUSlice[ihistoRU]->getTH1()->GetXaxis()->GetXmin(),
                                                      meBtlHitMultSMRUSlice[ihistoRU]->getTH1()->GetXaxis()->GetXmax());
  }

  // --- Calculate the cumulative histograms
  double bin_sum = meBtlHitLogEnergy->getBinContent(meBtlHitLogEnergy->getNbinsX() + 1);
  for (int ibin = meBtlHitLogEnergy->getNbinsX(); ibin >= 1; --ibin) {
    bin_sum += meBtlHitLogEnergy->getBinContent(ibin);
    meHitOccupancy_->setBinContent(ibin, scale_Crystals * bin_sum);
  }
  for(unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    double bin_sum_RUSlice = meBtlHitLogEnergyRUSlice[ihistoRU]->getBinContent(meBtlHitLogEnergyRUSlice[ihistoRU]->getNbinsX() + 1);
    for (int ibin = meBtlHitLogEnergyRUSlice[ihistoRU]->getNbinsX(); ibin >= 1; --ibin) {
      bin_sum_RUSlice += meBtlHitLogEnergyRUSlice[ihistoRU]->getBinContent(ibin);
      meHitOccupancyRUSlice_[ihistoRU]->setBinContent(ibin, scale_Crystals_RU * bin_sum_RUSlice);
    }
  }
  // --- Calculate the occupancy histograms
  for (int ibin = 0; ibin < meBtlHitMultCell->getNbinsX(); ibin++) {
    double bin_content = meBtlHitMultCell->getBinContent(ibin);
    meHitOccupancyCell_->setBinContent(ibin, bin_content * scale_Crystals);
  }
  for (int ibin = 0; ibin < meBtlHitMultSM->getNbinsX(); ibin++) {
    double bin_content = meBtlHitMultSM->getBinContent(ibin);
    meHitOccupancySM_->setBinContent(ibin, bin_content * scale_SMs);
  }
  for(unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    for (int ibin = 0; ibin < meBtlHitMultCellRUSlice[ihistoRU]->getNbinsX(); ibin ++) {
      double bin_content_RUSlice = meBtlHitMultCellRUSlice[ihistoRU]->getBinContent(ibin);
      meHitOccupancyCellRUSlice_[ihistoRU]->setBinContent(ibin, bin_content_RUSlice * scale_Crystals_RU);
    }
    for (int ibin = 0; ibin < meBtlHitMultSMRUSlice[ihistoRU]->getNbinsX(); ibin++) {
      double bin_content_RUSlice = meBtlHitMultSMRUSlice[ihistoRU]->getBinContent(ibin);
      meHitOccupancySMRUSlice_[ihistoRU]->setBinContent(ibin, bin_content_RUSlice * scale_SMs_RU);
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlSimHitsHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/SimHits/");

  descriptions.add("btlSimHitsPostProcessor", desc);
}

DEFINE_FWK_MODULE(BtlSimHitsHarvester);
