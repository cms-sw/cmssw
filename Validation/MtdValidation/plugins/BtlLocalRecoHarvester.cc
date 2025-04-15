// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlLocalRecoHarvester
//
/**\class BtlLocalRecoHarvester BtlLocalRecoHarvester.cc Validation/MtdValidation/plugins/BtlLocalRecoHarvester.cc

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

class BtlLocalRecoHarvester : public DQMEDHarvester {
public:
  explicit BtlLocalRecoHarvester(const edm::ParameterSet& iConfig);
  ~BtlLocalRecoHarvester() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  const std::string folder_;

  // --- Histograms
  MonitorElement* meHitOccupancyLog_;
  MonitorElement* meHitOccupancy_;
  static constexpr int nRU_ = BTLDetId::kRUPerRod;
  MonitorElement* meHitOccupancyRUSlice_[nRU_];
};

// ------------ constructor and destructor --------------
BtlLocalRecoHarvester::BtlLocalRecoHarvester(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {}

BtlLocalRecoHarvester::~BtlLocalRecoHarvester() {}

// ------------ endjob tasks ----------------------------
void BtlLocalRecoHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& igetter) {
  // --- Get the monitoring histograms
  MonitorElement* meBtlHitLogEnergy = igetter.get(folder_ + "BtlHitLogEnergy");
  MonitorElement* meBtlHitEnergy = igetter.get(folder_ + "BtlHitEnergy");
  MonitorElement* meBtlHitEnergyRUSlice[nRU_];
  MonitorElement* meNevents = igetter.get(folder_ + "BtlNevents");
  bool missing_ru_slice = false;
  for (unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    meBtlHitEnergyRUSlice[ihistoRU] = igetter.get(folder_ + "BtlHitEnergyRUSlice" + std::to_string(ihistoRU));
    if (!meBtlHitEnergyRUSlice[ihistoRU]) {
      missing_ru_slice = true;
    }
  }

  if (!meBtlHitLogEnergy || !meBtlHitEnergy || missing_ru_slice || !meNevents) {
    edm::LogError("BtlLocalRecoHarvester") << "Monitoring histograms not found!" << std::endl;
    return;
  }

  // --- Get the number of BTL crystals and the number of processed events
  const float NBtlCrystals = BTLDetId::kCrystalsBTL;
  const float Nevents = meNevents->getEntries();
  const float scale_Crystals = (Nevents > 0 ? 1. / (Nevents * NBtlCrystals) : 1.);
  const float scale_Crystals_RU = (Nevents > 0 ? 1. / (Nevents * NBtlCrystals / nRU_) : 1.);

  // --- Book the cumulative histograms
  ibook.cd(folder_);
  meHitOccupancyLog_ = ibook.book1D("BtlHitOccupancyLog",
                                 "BTL cell occupancy vs RECO hit energy;log_{10}(E_{RECO} [MeV]); Occupancy per event",
                                 meBtlHitLogEnergy->getNbinsX(),
                                 meBtlHitLogEnergy->getTH1()->GetXaxis()->GetXmin(),
                                 meBtlHitLogEnergy->getTH1()->GetXaxis()->GetXmax());
  meHitOccupancy_ = ibook.book1D("BtlHitOccupancy",
                                 "BTL cell occupancy vs RECO hit energy;E_{RECO} [MeV]; Occupancy per event",
                                 meBtlHitEnergy->getNbinsX(),
                                 meBtlHitEnergy->getTH1()->GetXaxis()->GetXmin(),
                                 meBtlHitEnergy->getTH1()->GetXaxis()->GetXmax());
  for(unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    std::string name_Energy = "BtlHitOccupancyRUSlice" + std::to_string(ihistoRU);
    std::string title_Energy = "BTL cell occupancy vs RECO hit energy (RU " + std::to_string(ihistoRU) + ");E_{RECO} [MeV]; Occupancy per event";
    meHitOccupancyRUSlice_[ihistoRU] = ibook.book1D(name_Energy,
                                                    title_Energy,
                                                    meBtlHitEnergyRUSlice[ihistoRU]->getNbinsX(),
                                                    meBtlHitEnergyRUSlice[ihistoRU]->getTH1()->GetXaxis()->GetXmin(),
                                                    meBtlHitEnergyRUSlice[ihistoRU]->getTH1()->GetXaxis()->GetXmax());
  }

  // --- Calculate the cumulative histograms
  double bin_sum_log = meBtlHitLogEnergy->getBinContent(meBtlHitLogEnergy->getNbinsX() + 1);
  for (int ibin = meBtlHitLogEnergy->getNbinsX(); ibin >= 1; --ibin) {
    bin_sum_log += meBtlHitLogEnergy->getBinContent(ibin);
    meHitOccupancyLog_->setBinContent(ibin, scale_Crystals * bin_sum_log);
  }
  double bin_sum = meBtlHitEnergy->getBinContent(meBtlHitEnergy->getNbinsX() + 1);
  for (int ibin = meBtlHitEnergy->getNbinsX(); ibin >= 1; --ibin) {
    bin_sum += meBtlHitEnergy->getBinContent(ibin);
    meHitOccupancy_->setBinContent(ibin, scale_Crystals * bin_sum);
  }
  for(unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    double bin_sum_RUSlice = meBtlHitEnergyRUSlice[ihistoRU]->getBinContent(meBtlHitEnergyRUSlice[ihistoRU]->getNbinsX() + 1);
    for (int ibin = meBtlHitEnergyRUSlice[ihistoRU]->getNbinsX(); ibin >= 1; --ibin) {
      bin_sum_RUSlice += meBtlHitEnergyRUSlice[ihistoRU]->getBinContent(ibin);
      meHitOccupancyRUSlice_[ihistoRU]->setBinContent(ibin, scale_Crystals_RU * bin_sum_RUSlice);
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlLocalRecoHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/LocalReco/");

  descriptions.add("btlLocalRecoPostProcessor", desc);
}

DEFINE_FWK_MODULE(BtlLocalRecoHarvester);
