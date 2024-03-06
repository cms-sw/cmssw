#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHostCollection.h"

/**
 * @short consumes channel digis and packs then in a realistic FEDRawData stream
 */
class HGCalDigiSoaFiller : public edm::stream::EDProducer<> {

public:

/**
   @short constructor
*/
  explicit HGCalDigiSoaFiller(edm::ParameterSet const& config)
    : digisCEET_( consumes<edm::SortedCollection<HGCalDataFrame> >(config.getParameter<edm::InputTag>("eeChannelDigis") ) ),
      digisCEHSiT_( consumes<edm::SortedCollection<HGCalDataFrame> >(config.getParameter<edm::InputTag>("cehsiChannelDigis") ) ),
      digisCEHSiPMT_( consumes<edm::SortedCollection<HGCalDataFrame> >(config.getParameter<edm::InputTag>("cehsipmChannelDigis") ) ) {
    
    digiProdT_ = produces(config.getParameter<std::string>("digisProdName"));
    
  }

  /**
     @short parameters to be used with this plugin
  */
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("eeChannelDigis", edm::InputTag("simHGCalUnsuppressedDigis","EE"));
    desc.add<edm::InputTag>("cehsiChannelDigis", edm::InputTag("simHGCalUnsuppressedDigis","HEfront"));
    desc.add<edm::InputTag>("cehsipmChannelDigis", edm::InputTag("simHGCalUnsuppressedDigis","HEback"));
    desc.add<std::string>("digisProdName", "HGC");
    descriptions.addWithDefaultLabel(desc);
    
  }


private:

  /**
     @short steers the conversion to SOA
  */
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override {
    
    //FIXME
    int itSample = 2;
    
    //read "classic" channel digis
    auto const &digisCEE = iEvent.get(digisCEET_);
    //auto const &digisCEHSi = iEvent.get(digisCEHSiT_);
    //auto const &digisCEHSiPM = iEvent.get(digisCEHSiPMT_);
    
    //fill the SoA collection
    int finaldigi_size = 100; //FIXME : to be retrieved from Elec Mapping 
    hgcaldigi::HGCalDigiHostCollection digiProd(finaldigi_size,cms::alpakatools::host());
    for(auto d : digisCEE) {
      
      int i=0;
      uint32_t detid( d.id().rawId() );
      uint32_t eleid( detid );  //FIXME : from mapping
      
      //read digi (in-time sample only)
      uint32_t toa = d.sample(itSample).toa();
      uint32_t tctp = d.sample(itSample).mode() ? 3 : 0;
      uint32_t adcm1( d.sample(itSample-1).data() );
      uint32_t adc( tctp<3 ? d.sample(itSample).data() : 0 );
      uint32_t tot( tctp==3 ? d.sample(itSample).data() : 0 );
      
      std::cout << "Filling with " << eleid << " " << toa << " " << tctp << " " << adcm1 << " " << adc << " " << tot << std::endl;
      
      auto idigi = digiProd.view()[i];
      idigi.electronicsId() = eleid;
      idigi.tctp() = tctp;
      idigi.adcm1() = adcm1;
      idigi.adc() = adc;
      idigi.tot() = tot;
      idigi.toa() = toa;
      idigi.cm() = 0;
      idigi.flags() = 0;

      break;
    }
    
    
    iEvent.emplace(digiProdT_, std::move(digiProd));
  }
    
 
  edm::EDGetTokenT<edm::SortedCollection<HGCalDataFrame> > digisCEET_,digisCEHSiT_,digisCEHSiPMT_;
  edm::EDPutTokenT<hgcaldigi::HGCalDigiHostCollection> digiProdT_;
};

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalDigiSoaFiller);
