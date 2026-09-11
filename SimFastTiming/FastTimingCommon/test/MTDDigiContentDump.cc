#include <memory>
#include <iostream>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FTLDigi/interface/MTDDigiCollections.h"

class MTDDigiContentDump : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit MTDDigiContentDump(const edm::ParameterSet&);
  ~MTDDigiContentDump() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<BTLDigiContentCollection> tok_BTL_digi;
  edm::EDGetTokenT<ETLDigiContentCollection> tok_ETL_digi;
};

MTDDigiContentDump::MTDDigiContentDump(const edm::ParameterSet& iConfig)

{
  tok_BTL_digi = consumes<BTLDigiContentCollection>(edm::InputTag("mix", "MTDBarrel"));
  tok_ETL_digi = consumes<ETLDigiContentCollection>(edm::InputTag("mix", "MTDEndcap"));
}

MTDDigiContentDump::~MTDDigiContentDump() {}

//
// member functions
//

// ------------ method called for each event ------------
void MTDDigiContentDump::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;

  edm::Handle<BTLDigiContentCollection> h_BTL_digi;
  iEvent.getByToken(tok_BTL_digi, h_BTL_digi);

  edm::Handle<ETLDigiContentCollection> h_ETL_digi;
  iEvent.getByToken(tok_ETL_digi, h_ETL_digi);

  // --- BTL DIGIs:

  if (!h_BTL_digi->empty()) {
    std::cout << " ----------------------------------------" << std::endl;
    std::cout << " BTL DIGI collection: \n" << std::endl;

    for (const auto& btldigi : *h_BTL_digi) {
      std::cout << btldigi << std::endl;

    }  // digi loop

  }  // if ( h_BTL_digi->size() > 0 )

  // --- ETL DIGIs:

  if (!h_ETL_digi->empty()) {
    std::cout << "\n ----------------------------------------" << std::endl;
    std::cout << " ETL DIGI collection: \n" << std::endl;

    for (const auto& etldigi : *h_ETL_digi) {
      std::cout << etldigi << std::endl;

      // --- loop over the dataFrame samples
      //for (int isample = 0; isample < dataFrame.size(); ++isample) {
      //  const auto& sample = dataFrame.sample(isample);

      //  std::cout << "       sample " << isample << ":";
      //  if (sample.data() == 0 && sample.toa() == 0) {
      //    std::cout << std::endl;
      //    continue;
      //  }
      //  std::cout << "  amplitude = " << sample.data() << "  time = " << sample.toa() << " r/c = " << sample.row()
      //  << " / " << sample.column() << " th = " << sample.threshold() << " mode = " << sample.mode()
      //  << std::endl;

      //}  // isample loop

    }  // digi loop

  }  // if ( h_ETL_digi->size() > 0 )
}

// ------------ method called once each job just before starting event loop  ------------
void MTDDigiContentDump::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void MTDDigiContentDump::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MTDDigiContentDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDDigiContentDump);
