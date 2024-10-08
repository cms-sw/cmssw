// -*- C++ -*-
//
// Package:    TauAnalysis/MCEmbeddingTools
// Class:      EmbeddingDetecorStateFilter
//
/**\class EmbeddingDetectorStateFilter EmbeddingDetectorStateFilter.cc TauAnalysis/MCEmbeddingTools/plugins/EmbeddingDetectorStateFilter.cc

 Description: Dummy implementation of the DetectorStateFilter class from DQM/TrackerCommon/plugins/DetectorStateFilter.cc

 Implementation:
     [Notes on implementation]
*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <type_traits>  // for std::is_same


class EmbeddingDetectorStateFilter : public edm::stream::EDFilter<> {
public:
  EmbeddingDetectorStateFilter(const edm::ParameterSet&);
  ~EmbeddingDetectorStateFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, edm::EventSetup const&) override;

  const bool verbose_;
  uint64_t nEvents_, nSelectedEvents_;
  bool detectorOn_;
  const std::string detectorType_;
  const std::vector<std::string> combinations_;  // Vector of strings specifying accepted combinations
  const edm::EDGetTokenT<DcsStatusCollection> dcsStatusLabel_;
  const edm::EDGetTokenT<DCSRecord> dcsRecordToken_;
};


/**
 * Dummy implementation of the DetectorStateFilter class in DQM/TrackerCommon/plugins/DetectorStateFilter.cc.
 *
 * This implementation emulates the branch of the DetectorStateFilter class for MC events. Embedding events are data events, but the detector is simulated. This means that this module fails as the detector state of the simulated detector is tried to read out. This dummy implementation aims to mitigate the detector read-out so that this module, which is part of some HLT sequences, is passed.
 */
EmbeddingDetectorStateFilter::EmbeddingDetectorStateFilter(const edm::ParameterSet& pset)
    : verbose_(pset.getUntrackedParameter<bool>("DebugOn", false)),
      detectorType_(pset.getUntrackedParameter<std::string>("DetectorType", "sistrip")),
      combinations_(pset.getUntrackedParameter<std::vector<std::string>>("acceptedCombinations")),
      dcsStatusLabel_(consumes<DcsStatusCollection>(
          pset.getUntrackedParameter<edm::InputTag>("DcsStatusLabel", edm::InputTag("scalersRawToDigi")))),
      dcsRecordToken_(consumes<DCSRecord>(
          pset.getUntrackedParameter<edm::InputTag>("DCSRecordLabel", edm::InputTag("onlineMetaDataDigis")))) {
  nEvents_ = 0;
  nSelectedEvents_ = 0;
  detectorOn_ = false;
}


/**
 * Destructor of the EmbeddingDetectorStateFilter class.
 */
EmbeddingDetectorStateFilter::~EmbeddingDetectorStateFilter() = default;


/**
 * Dummy filter implementation, in which each event passes the filter.
 */
bool EmbeddingDetectorStateFilter::filter(edm::Event& evt, edm::EventSetup const& es)
{
  // implement the branch for MC events from the original DetectorStateFilter implementation
  nEvents_++;
  detectorOn_ = true;
  nSelectedEvents_++;
  if (verbose_) {
    edm::LogInfo("DetectorStatusFilter") << "Total MC Events " << nEvents_ << " Selected Events " << nSelectedEvents_
                                         << " Detector States " << detectorOn_ << std::endl;
  }
  return detectorOn_;
}


/**
 * Fill 'descriptions' with the allowed parameters for the module.
 */
void EmbeddingDetectorStateFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("filters on the HV status of the Tracker (either pixels or strips)");
  desc.addUntracked<bool>("DebugOn", false)->setComment("activates debugging");
  desc.addUntracked<std::string>("DetectorType", "sistrip")->setComment("either strips or pixels");
  desc.addUntracked<std::vector<std::string>>("acceptedCombinations", {});
  desc.addUntracked<edm::InputTag>("DcsStatusLabel", edm::InputTag("scalersRawToDigi"))
      ->setComment("event data for DCS (Run2)");
  desc.addUntracked<edm::InputTag>("DCSRecordLabel", edm::InputTag("onlineMetaDataDigis"))
      ->setComment("event data for DCS (Run3)");
  descriptions.add("_detectorStateFilter", desc);
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EmbeddingDetectorStateFilter);
