// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlDigiHitsValidation
//
/**\class BtlDigiHitsValidation BtlDigiHitsValidation.cc Validation/MtdValidation/plugins/BtlDigiHitsValidation.cc

 Description: BTL DIGI hits validation

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Massimo Casarsa
//         Created:  Mon, 11 Mar 2019 14:12:22 GMT
//
//

#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"


//
// class declaration
//


class BtlDigiHitsValidation : public DQMEDAnalyzer {

public:
  explicit BtlDigiHitsValidation(const edm::ParameterSet&);
  ~BtlDigiHitsValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  void bookHistograms(DQMStore::IBooker &,
		      edm::Run const&,
		      edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const MTDGeometry* geom_;
  const MTDTopology* topo_;

  const std::string folder_;
  const std::string infoLabel_;
  const std::string btlDigisCollection_;

  int eventCount_;

  edm::EDGetTokenT<BTLDigiCollection> btlDigiHitsToken_;

  MonitorElement* meHitCharge_[2];
  MonitorElement* meHitTime1_[2];
  MonitorElement* meHitTime2_[2];

  MonitorElement* meOccupancy_[2];

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BtlDigiHitsValidation::BtlDigiHitsValidation(const edm::ParameterSet& iConfig):
  geom_(nullptr),
  topo_(nullptr),
  folder_(iConfig.getParameter<std::string>("folder")),
  infoLabel_(iConfig.getParameter<std::string>("moduleLabel")),
  btlDigisCollection_(iConfig.getParameter<std::string>("btlDigiHitsCollection")),
  eventCount_(0) {

  btlDigiHitsToken_ = consumes <BTLDigiCollection> (edm::InputTag(std::string(infoLabel_),
								  std::string(btlDigisCollection_)));

}


BtlDigiHitsValidation::~BtlDigiHitsValidation() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called for each event  ------------
void BtlDigiHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;

  edm::LogInfo("EventInfo") << " Run = " << iEvent.id().run() << " Event = " << iEvent.id().event();

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  geom_ = geometryHandle.product();

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);
  topo_ = topologyHandle.product();

  edm::Handle<BTLDigiCollection> btlDigiHitsHandle;
  iEvent.getByToken(btlDigiHitsToken_, btlDigiHitsHandle);

  if( ! btlDigiHitsHandle.isValid() ) return;

  eventCount_++;
  
  // --- Loop over the BLT DIGI hits
  for (const auto& dataFrame: *btlDigiHitsHandle) {

    //DetId id =  dataFrame.id();
      
    const auto& sample_L = dataFrame.sample(0);
    const auto& sample_R = dataFrame.sample(1);

    uint32_t adcL  = sample_L.data();
    uint32_t adcR  = sample_R.data();
    uint32_t tdc1L = sample_L.toa();
    uint32_t tdc1R = sample_R.toa();
    uint32_t tdc2L = sample_L.toa2();
    uint32_t tdc2R = sample_R.toa2();

    meHitCharge_[0]->Fill(adcL); 
    meHitCharge_[1]->Fill(adcR); 
    meHitTime1_[0]->Fill(tdc1L);
    meHitTime1_[1]->Fill(tdc1R);
    meHitTime2_[0]->Fill(tdc2L);
    meHitTime2_[1]->Fill(tdc2R);

  } // dataFrame loop



}

void BtlDigiHitsValidation::bookHistograms(DQMStore::IBooker & ibook,
                               edm::Run const& run,
                               edm::EventSetup const & iSetup) {

  ibook.setCurrentFolder(folder_);

  meHitCharge_[0] = ibook.book1D("BtlHitChargeL", "BTL DIGI hits charge (L);amplitude [ADC counts]",
				 1024, 0., 1024.);
  meHitCharge_[1] = ibook.book1D("BtlHitChargeR", "BTL DIGI hits charge (R);amplitude [ADC counts]",
				 1024, 0., 1024.);
  meHitTime1_[0]  = ibook.book1D("BtlHitTime1L", "BTL DIGI hits ToA1 (L);ToA [TDC counts]", 
				 1024, 0., 1024.);
  meHitTime1_[1]  = ibook.book1D("BtlHitTime1R", "BTL DIGI hits ToA1 (R);ToA [TDC counts]", 
				 1024, 0., 1024.);
  meHitTime2_[0]  = ibook.book1D("BtlHitTime2L", "BTL DIGI hits ToA2 (L);ToA [TDC counts]", 
				 1024, 0., 1024.);
  meHitTime2_[1]  = ibook.book1D("BtlHitTime2R", "BTL DIGI hits ToA2 (R);ToA [TDC counts]", 
				 1024, 0., 1024.);
  meOccupancy_[0] = ibook.book2D("BtlOccupancyL","BTL DIGI hits occupancy (L);z [cm]; #phi [rad]",
				 65, -260., 260., 315, -3.15, 3.15 );
  meOccupancy_[1] = ibook.book2D("BtlOccupancyR","BTL DIGI hits occupancy (R);z [cm]; #phi [rad]",
				 65, -260., 260., 315, -3.15, 3.15 );

}



// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlDigiHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no
  // validation
  // Please change this to state exactly what you do use, even if it is no
  // parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "MTD/BTL/DigiHits");
  desc.add<std::string>("moduleLabel","mix");
  desc.add<std::string>("btlDigiHitsCollection","FTLBarrel");
  descriptions.add("btlDigiHits", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BtlDigiHitsValidation);
