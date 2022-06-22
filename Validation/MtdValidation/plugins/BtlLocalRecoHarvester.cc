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
  MonitorElement* meHitOccupancy_;
};

// ------------ constructor and destructor --------------
BtlLocalRecoHarvester::BtlLocalRecoHarvester(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {}

BtlLocalRecoHarvester::~BtlLocalRecoHarvester() {}

// ------------ endjob tasks ----------------------------
void BtlLocalRecoHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& igetter) {
  // --- Get the monitoring histograms
  MonitorElement* meBtlHitLogEnergy = igetter.get(folder_ + "BtlHitLogEnergy");
  MonitorElement* meNevents = igetter.get(folder_ + "BtlNevents");

  if (!meBtlHitLogEnergy || !meNevents) {
    edm::LogError("BtlLocalRecoHarvester") << "Monitoring histograms not found!" << std::endl;
    return;
  }

  // --- Get the number of BTL crystals and the number of processed events
  const float NBtlCrystals = BTLDetId::kCrystalsPerRODBarPhiFlat * BTLDetId::MAX_ROD;
  const float Nevents = meNevents->getEntries();
  const float scale = (Nevents > 0 ? 1. / (Nevents * NBtlCrystals) : 1.);

  // --- Book the cumulative histogram
  ibook.cd(folder_);
  meHitOccupancy_ = ibook.book1D("BtlHitOccupancy",
                                 "BTL cell occupancy vs RECO hit energy;log_{10}(E_{RECO} [MeV]); Occupancy per event",
                                 meBtlHitLogEnergy->getNbinsX(),
                                 meBtlHitLogEnergy->getTH1()->GetXaxis()->GetXmin(),
                                 meBtlHitLogEnergy->getTH1()->GetXaxis()->GetXmax());

  // --- Calculate the cumulative histogram
  double bin_sum = meBtlHitLogEnergy->getBinContent(meBtlHitLogEnergy->getNbinsX() + 1);
  for (int ibin = meBtlHitLogEnergy->getNbinsX(); ibin >= 1; --ibin) {
    bin_sum += meBtlHitLogEnergy->getBinContent(ibin);
    meHitOccupancy_->setBinContent(ibin, scale * bin_sum);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlLocalRecoHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/LocalReco/");

  descriptions.add("btlLocalRecoPostProcessor", desc);
}

DEFINE_FWK_MODULE(BtlLocalRecoHarvester);
