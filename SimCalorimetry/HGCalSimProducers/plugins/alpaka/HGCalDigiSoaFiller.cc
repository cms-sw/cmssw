#include <cstdio>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHostCollection.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDeviceCollection.h"
#include "CondFormats/DataRecord/interface/HGCalMappingModuleRcd.h"
#include "CondFormats/DataRecord/interface/HGCalMappingCellRcd.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDeviceCollection.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  /**
   *   @short consumes channel digis and packs then in a realistic FEDRawData stream
  */ 
  class HGCalDigiSoaFiller : public stream::EDProducer<> {

  public:
    using CellInfo = hgcal::HGCalMappingCellParamDeviceCollection;
    using ModuleInfo = hgcal::HGCalMappingModuleParamDeviceCollection;

    /**
      @  short constructor
    */
    explicit HGCalDigiSoaFiller(edm::ParameterSet const& config);

   /**
       @short parameters to be used with this plugin
    */
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions); 

    /**
      @short steers the conversion to SOA
    */
    void produce(device::Event&, device::EventSetup const&) override;
  
  private: 

    edm::EDGetTokenT<edm::SortedCollection<HGCalDataFrame> > digisCEET_,digisCEHSiT_,digisCEHSiPMT_;
    device::EDPutToken<hgcaldigi::HGCalDigiDeviceCollection> digiProdT_;

    device::ESGetToken<CellInfo, HGCalMappingCellRcd> cellTkn_;
    device::ESGetToken<ModuleInfo, HGCalMappingModuleRcd> moduleTkn_;
  };


  //
  HGCalDigiSoaFiller::HGCalDigiSoaFiller(const edm::ParameterSet& config)
       : digisCEET_( consumes<edm::SortedCollection<HGCalDataFrame> >(config.getParameter<edm::InputTag>("eeChannelDigis") ) ),
         digisCEHSiT_( consumes<edm::SortedCollection<HGCalDataFrame> >(config.getParameter<edm::InputTag>("cehsiChannelDigis") ) ),
         digisCEHSiPMT_( consumes<edm::SortedCollection<HGCalDataFrame> >(config.getParameter<edm::InputTag>("cehsipmChannelDigis") ) ),
         cellTkn_(esConsumes()),
         moduleTkn_(esConsumes()) {
      digiProdT_ = produces(config.getParameter<std::string>("digisProdName"));         
  }

  //
  void HGCalDigiSoaFiller::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
     edm::ParameterSetDescription desc;
     desc.add<edm::InputTag>("eeChannelDigis", edm::InputTag("simHGCalUnsuppressedDigis","EE"));
     desc.add<edm::InputTag>("cehsiChannelDigis", edm::InputTag("simHGCalUnsuppressedDigis","HEfront"));
     desc.add<edm::InputTag>("cehsipmChannelDigis", edm::InputTag("simHGCalUnsuppressedDigis","HEback"));
     desc.add<std::string>("digisProdName", "HGC");
     descriptions.addWithDefaultLabel(desc);
    }

  //
  void HGCalDigiSoaFiller::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    
    const ModuleInfo& modules = iSetup.getData(moduleTkn_);
    const CellInfo& cells = iSetup.getData(cellTkn_);


    //FIXME
    int itSample = 2;
    
    //read "classic" channel digis
    auto const &digisCEE = iEvent.get(digisCEET_);
    //auto const &digisCEHSi = iEvent.get(digisCEHSiT_);
    //auto const &digisCEHSiPM = iEvent.get(digisCEHSiPMT_);
    
    //fill the SoA collection in the host
    int finaldigi_size = 100; //FIXME : to be retrieved from Elec Mapping 
    hgcaldigi::HGCalDigiHostCollection host_buffer(finaldigi_size,cms::alpakatools::host());
    for(auto d : digisCEE) {
      
      int i=0;
      uint32_t detid( d.id().rawId() );
      uint32_t eleid = ::hgcal::mappingtools::getElectronicsIdForSiCell<ModuleInfo,CellInfo>(modules,cells,detid);
      
      //read digi (in-time sample only)
      uint32_t toa = d.sample(itSample).toa();
      uint32_t tctp = d.sample(itSample).mode() ? 3 : 0;
      uint32_t adcm1( d.sample(itSample-1).data() );
      uint32_t adc( tctp<3 ? d.sample(itSample).data() : 0 );
      uint32_t tot( tctp==3 ? d.sample(itSample).data() : 0 );
      
      std::cout << "Filling with " << eleid << " " << toa << " " << tctp << " " << adcm1 << " " << adc << " " << tot << std::endl;
      
      auto idigi = host_buffer.view()[i];
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
    
    //allocate device colleciton, copy from host and put in event
    auto& queue = iEvent.queue();
    hgcaldigi::HGCalDigiDeviceCollection device_buffer(host_buffer.view().metadata().size(),queue);
    alpaka::memcpy(queue, host_buffer.buffer(), device_buffer.const_buffer());
    iEvent.emplace(digiProdT_, std::move(device_buffer));
  }

  

}//end ALPAKA_ACCELERATOR_NAMESPACE

// define this as a plug-in
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalDigiSoaFiller);
