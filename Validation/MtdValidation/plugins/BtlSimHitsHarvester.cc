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
  static constexpr int nRU_ = 6;
  MonitorElement* meHitOccupancyRUSlice_[nRU_];
};

// ------------ constructor and destructor --------------
BtlSimHitsHarvester::BtlSimHitsHarvester(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {}

BtlSimHitsHarvester::~BtlSimHitsHarvester() {}

// ------------ endjob tasks ----------------------------
void BtlSimHitsHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& igetter) {
  // --- Get the monitoring histograms
  MonitorElement* meBtlHitLogEnergy = igetter.get(folder_ + "BtlHitLogEnergy");
  MonitorElement* meNevents = igetter.get(folder_ + "BtlNevents");
  MonitorElement* meBtlHitLogEnergyRUSlice[nRU_];
  bool missing_ru_slice = false;
  for(unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    meBtlHitLogEnergyRUSlice[ihistoRU] = igetter.get(folder_ + "BtlHitLogEnergyRUSlice_" + std::to_string(ihistoRU + 1));
    if(!meBtlHitLogEnergyRUSlice[ihistoRU]){
      missing_ru_slice = true;
    }
  }
  if (!meBtlHitLogEnergy || !meNevents || missing_ru_slice) {
    edm::LogError("BtlSimHitsHarvester") << "Monitoring histograms not found!" << std::endl;
    return;
  }

  // --- Get the number of BTL crystals and the number of processed events
  const float NBtlCrystals = BTLDetId::kCrystalsBTL;
  const float Nevents = meNevents->getEntries();
  const float scale = (Nevents > 0 ? 1. / (Nevents * NBtlCrystals) : 1.);

  // --- Book the cumulative histogram
  ibook.cd(folder_);
  meHitOccupancy_ = ibook.book1D("BtlHitOccupancy",
                                 "BTL cell occupancy vs hit energy;log_{10}(E_{SIM} [MeV]); Occupancy per event",
                                 meBtlHitLogEnergy->getNbinsX(),
                                 meBtlHitLogEnergy->getTH1()->GetXaxis()->GetXmin(),
                                 meBtlHitLogEnergy->getTH1()->GetXaxis()->GetXmax());
  for(unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    std::string name = "BtlHitOccupancyRUSlice" + std::to_string(ihistoRU + 1);
    std::string title = "BTL cell occupancy vs hit energy (RU " + std::to_string(ihistoRU + 1) + ");log_{10}(E_{SIM} [MeV]); Occupancy per event";
    meHitOccupancyRUSlice_[ihistoRU] = ibook.book1D(name,
                                                    title,
                                                    meBtlHitLogEnergyRUSlice[ihistoRU]->getNbinsX(),
                                                    meBtlHitLogEnergyRUSlice[ihistoRU]->getTH1()->GetXaxis()->GetXmin(),
                                                    meBtlHitLogEnergyRUSlice[ihistoRU]->getTH1()->GetXaxis()->GetXmax());
    }

  // --- Calculate the cumulative histogram
  double bin_sum = meBtlHitLogEnergy->getBinContent(meBtlHitLogEnergy->getNbinsX() + 1);
  for (int ibin = meBtlHitLogEnergy->getNbinsX(); ibin >= 1; --ibin) {
    bin_sum += meBtlHitLogEnergy->getBinContent(ibin);
    meHitOccupancy_->setBinContent(ibin, scale * bin_sum);
  }
  for(unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    double bin_sum_RUSlice = meBtlHitLogEnergyRUSlice[ihistoRU]->getBinContent(meBtlHitLogEnergyRUSlice[ihistoRU]->getNbinsX() + 1);
    for (int ibin = meBtlHitLogEnergyRUSlice[ihistoRU]->getNbinsX(); ibin >= 1; --ibin) {
      bin_sum_RUSlice += meBtlHitLogEnergyRUSlice[ihistoRU]->getBinContent(ibin);
      meHitOccupancyRUSlice_[ihistoRU]->setBinContent(ibin, scale * bin_sum_RUSlice);
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
