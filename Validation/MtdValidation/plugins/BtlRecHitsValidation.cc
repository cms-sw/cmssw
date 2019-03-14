// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlRecHitsValidation
//
/**\class BtlRecHitsValidation BtlRecHitsValidation.cc Validation/MtdValidation/plugins/BtlRecHitsValidation.cc

 Description: BTL RECO hits validation

 Implementation:
     [Notes on implementation]
*/

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"


class BtlRecHitsValidation : public DQMEDAnalyzer {

public:
  explicit BtlRecHitsValidation(const edm::ParameterSet&);
  ~BtlRecHitsValidation() override;

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
  const std::string btlRecHitsCollection_;

  int eventCount_;

  edm::EDGetTokenT<FTLRecHitCollection> btlRecHitsToken_;

  // --- histograms declaration

  MonitorElement* meHitEnergy_;
  MonitorElement* meHitTime_;

  MonitorElement* meOccupancy_;

};


// ------------ constructor and destructor --------------
BtlRecHitsValidation::BtlRecHitsValidation(const edm::ParameterSet& iConfig):
  geom_(nullptr),
  topo_(nullptr),
  folder_(iConfig.getParameter<std::string>("folder")),
  infoLabel_(iConfig.getParameter<std::string>("moduleLabel")),
  btlRecHitsCollection_(iConfig.getParameter<std::string>("btlRecHitsCollection")),
  eventCount_(0) {

  btlRecHitsToken_ = consumes <FTLRecHitCollection> (edm::InputTag(std::string(infoLabel_),
								   std::string(btlRecHitsCollection_)));

}

BtlRecHitsValidation::~BtlRecHitsValidation() {
}


// ------------ method called for each event  ------------
void BtlRecHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;

  edm::LogInfo("EventInfo") << " Run = " << iEvent.id().run() << " Event = " << iEvent.id().event();

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  geom_ = geometryHandle.product();

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);
  topo_ = topologyHandle.product();

  edm::Handle<FTLRecHitCollection> btlRecHitsHandle;
  iEvent.getByToken(btlRecHitsToken_, btlRecHitsHandle);

  if( ! btlRecHitsHandle.isValid() ) {
    edm::LogWarning("DataNotFound") << "No BTL RecHits found";
    return;
  }

  eventCount_++;

  // --- Loop over the BLT RECO hits
  for (const auto& recHit: *btlRecHitsHandle) {

    BTLDetId detId = recHit.id();
    DetId geoId = detId.geographicalId( static_cast<BTLDetId::CrysLayout>(topo_->getMTDTopologyMode()) );
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if( thedet == nullptr )
      throw cms::Exception("BtlRecHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId()
						   << " (" << detId.rawId()<< ") is invalid!" << std::dec
						   << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    Local3DPoint local_point(0.,0.,0.);
    local_point = topo.pixelToModuleLocalPoint(local_point,detId.row(topo.nrows()),detId.column(topo.nrows()));
    const auto& global_point = thedet->toGlobal(local_point);

    meHitEnergy_->Fill(recHit.energy());
    meHitTime_->Fill(recHit.time());

    meOccupancy_->Fill(global_point.z(),global_point.phi());

  } // recHit loop

}


// ------------ method for histogram booking ------------
void BtlRecHitsValidation::bookHistograms(DQMStore::IBooker & ibook,
                               edm::Run const& run,
                               edm::EventSetup const & iSetup) {

  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meHitEnergy_ = ibook.book1D("BtlHitEnergy", "BTL RECO hits energy;energy [MeV]", 200, 0., 20.);
  meHitTime_ = ibook.book1D("BtlHitTime", "BTL RECO hits ToA;ToA [ns]", 250, 0., 25.);

  meOccupancy_ = ibook.book2D("BtlOccupancy","BTL RECO hits occupancy;z [cm]; #phi [rad]",
			      65, -260., 260., 315, -3.15, 3.15 );

}



// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlRecHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/RecHits");
  desc.add<std::string>("moduleLabel","mtdRecHits");
  desc.add<std::string>("btlRecHitsCollection","FTLBarrel");

  descriptions.add("btlRecHits", desc);

}

DEFINE_FWK_MODULE(BtlRecHitsValidation);
