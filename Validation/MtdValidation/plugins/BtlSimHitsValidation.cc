// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlSimHitsValidation
//
/**\class BtlSimHitsValidation BtlSimHitsValidation.cc Validation/MtdValidation/plugins/BtlSimHitsValidation.cc

 Description: BTL SIM hits validation

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

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"


struct MTDHit {
  float energy;
  float time;
  float x;
  float y;
  float z;
};


class BtlSimHitsValidation : public DQMEDAnalyzer {

public:
  explicit BtlSimHitsValidation(const edm::ParameterSet&);
  ~BtlSimHitsValidation() override;

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
  const std::string g4InfoLabel_;
  const std::string btlHitsCollection_;

  const float hitMinEnergy_;

  int eventCount_;

  edm::EDGetTokenT<edm::PSimHitContainer> btlSimHitsToken_;

  // --- histograms declaration

  MonitorElement* meHitEnergy_;
  MonitorElement* meHitTime_;

  MonitorElement* meOccupancy_;

};


// ------------ constructor and destructor --------------
BtlSimHitsValidation::BtlSimHitsValidation(const edm::ParameterSet& iConfig):
  geom_(nullptr),
  topo_(nullptr),
  folder_(iConfig.getParameter<std::string>("folder")),
  g4InfoLabel_(iConfig.getParameter<std::string>("moduleLabelG4")),
  btlHitsCollection_(iConfig.getParameter<std::string>("btlSimHitsCollection")),
  hitMinEnergy_( iConfig.getParameter<double>("hitMinimumEnergy") ),
  eventCount_(0) {

  btlSimHitsToken_ = consumes <edm::PSimHitContainer> (edm::InputTag(std::string(g4InfoLabel_),
								     std::string(btlHitsCollection_)));

}

BtlSimHitsValidation::~BtlSimHitsValidation() {
}


// ------------ method called for each event  ------------
void BtlSimHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;

  edm::LogInfo("EventInfo") << " Run = " << iEvent.id().run() << " Event = " << iEvent.id().event();

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  geom_ = geometryHandle.product();

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);
  topo_ = topologyHandle.product();

  edm::Handle<edm::PSimHitContainer> btlSimHitsHandle;
  iEvent.getByToken(btlSimHitsToken_, btlSimHitsHandle);

  if( ! btlSimHitsHandle.isValid() ) {
    edm::LogWarning("DataNotFound") << "No BTL SIM hits found";
    return;
  }

  eventCount_++;
  
  std::map<uint32_t, MTDHit> m_btlHits;

  // --- Loop over the BLT SIM hits
  for (auto const& simHit: *btlSimHitsHandle) {

    // --- Use only hits compatible with the in-time bunch-crossing
    if ( simHit.tof() < 0 || simHit.tof() > 25. ) continue;

    DetId id = simHit.detUnitId();

    auto simHitIt = m_btlHits.emplace(id.rawId(),MTDHit()).first;

    // --- Accumulate the energy (in MeV) of SIM hits in the same detector cell
    (simHitIt->second).energy += 1000.*simHit.energyLoss();

    // --- Get the time of the first SIM hit in the cell
    if( (simHitIt->second).time==0 || simHit.tof()<(simHitIt->second).time ) {

      (simHitIt->second).time = simHit.tof();

      auto hit_pos = simHit.entryPoint();
      (simHitIt->second).x = hit_pos.x();
      (simHitIt->second).y = hit_pos.y();
      (simHitIt->second).z = hit_pos.z();

    }

  } // simHit loop


  // ==============================================================================
  //  Histogram filling
  // ==============================================================================

  for (auto const& hit: m_btlHits) {

    if ( (hit.second).energy < hitMinEnergy_ ) continue;

    // --- Get the SIM hit global position
    BTLDetId detId(hit.first); 
    DetId geoId = detId.geographicalId( static_cast<BTLDetId::CrysLayout>(topo_->getMTDTopologyMode()) );
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if( thedet == nullptr )
      throw cms::Exception("BtlSimHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId()
						   << " (" << detId.rawId()<< ") is invalid!" << std::dec
						   << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    Local3DPoint local_point(0.1*(hit.second).x,0.1*(hit.second).y,0.1*(hit.second).z);
    local_point = topo.pixelToModuleLocalPoint(local_point,detId.row(topo.nrows()),detId.column(topo.nrows()));
    const auto& global_point = thedet->toGlobal(local_point);

    // --- Fill the histograms
    meHitEnergy_->Fill((hit.second).energy);
    meHitTime_->Fill((hit.second).time);
    meOccupancy_->Fill(global_point.z(),global_point.phi());

  } // hit loop

}


// ------------ method for histogram booking ------------
void BtlSimHitsValidation::bookHistograms(DQMStore::IBooker & ibook,
                               edm::Run const& run,
                               edm::EventSetup const & iSetup) {

  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meHitEnergy_ = ibook.book1D("BtlHitEnergy", "BTL SIM hits energy;E_{SIM} [MeV]", 200, 0., 20.);
  meHitTime_   = ibook.book1D("BtlHitTime", "BTL SIM hits ToA;ToA_{SIM} [ns]", 250, 0., 25.);

  meOccupancy_ = ibook.book2D("BtlOccupancy","BTL SIM hits occupancy;z_{SIM} [cm];#phi_{SIM} [rad]",
			      520, -260., 260., 315, -3.15, 3.15 );

}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlSimHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/SimHits");
  desc.add<std::string>("moduleLabelG4","g4SimHits");
  desc.add<std::string>("btlSimHitsCollection","FastTimerHitsBarrel");
  desc.add<double>("hitMinimumEnergy",1.); // [MeV]

  descriptions.add("btlSimHits", desc);

}

DEFINE_FWK_MODULE(BtlSimHitsValidation);
