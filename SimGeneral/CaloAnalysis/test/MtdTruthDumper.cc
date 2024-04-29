// system include files
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValidHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"

class MtdTruthDumper : public edm::one::EDAnalyzer<> {
public:
  explicit MtdTruthDumper(const edm::ParameterSet&);
  ~MtdTruthDumper() override{};

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override{};
  void endJob() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<MtdSimLayerClusterCollection> mtdSimLCToken_;
};

MtdTruthDumper::MtdTruthDumper(const edm::ParameterSet& iConfig)
    : mtdSimLCToken_(
          consumes<MtdSimLayerClusterCollection>(iConfig.getParameter<edm::InputTag>("moduleLabelMtdSimLC"))) {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MtdTruthDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto mtdSimLCcoll = edm::makeValid(iEvent.getHandle(mtdSimLCToken_));

  edm::LogPrint("DumpMtdSimLC") << "\n MtdSimLayerCluster collection dump \n";
  edm::LogPrint("DumpMtdSimLC") << " MtdSimLayerCluster in the event = " << (*mtdSimLCcoll).size();
  size_t isimLC(0);

  isimLC = 0;
  for (const auto& mtdLC : *mtdSimLCcoll) {
    edm::LogPrint("DumpMtdSimLC") << "MtdSimLayerCluster " << isimLC << " = " << mtdLC;
    size_t ihit(0);
    for (const auto& hit : mtdLC.detIds_and_rows()) {
      edm::LogPrint("DumpMtdSimLC") << "hit # " << ihit << " DetId " << hit.first << " r/c "
                                    << (uint32_t)hit.second.first << " " << (uint32_t)hit.second.second;
      ihit++;
    }
    isimLC++;
    edm::LogPrint("DumpMtdSimLC") << "\n";
  }

  return;
}

void MtdTruthDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("moduleLabelMtdSimLC", edm::InputTag("mix", "MergedMtdTruthLC"))
      ->setComment("Module for input MtdSimLayerCluster collection");
  descriptions.add("mtdTruthDumper", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(MtdTruthDumper);
