// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      EtlSimHitsValidation
//
/**\class EtlSimHitsValidation EtlSimHitsValidation.cc Validation/MtdValidation/plugins/EtlSimHitsValidation.cc

 Description: ETL SIM hits validation

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Massimo Casarsa
//         Created:  Mon, 11 Mar 2019 13:56:22 GMT
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

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"


//
// class declaration
//


class EtlSimHitsValidation : public DQMEDAnalyzer {

public:
  explicit EtlSimHitsValidation(const edm::ParameterSet&);
  ~EtlSimHitsValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  void bookHistograms(DQMStore::IBooker &,
		      edm::Run const&,
		      edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

// ------------ member data ------------

  const MTDGeometry* geom_;

  const std::string folder_;
  const std::string g4InfoLabel_;
  const std::string etlHitsCollection_;

  const float hitMinEnergy_;

  int eventCount_;

  edm::EDGetTokenT<edm::PSimHitContainer> etlSimHitsToken_;

  MonitorElement* meHitEnergyZpos_;
  MonitorElement* meHitEnergyZneg_;
  MonitorElement* meHitTimeZpos_;
  MonitorElement* meHitTimeZneg_;

  MonitorElement* meOccupancyZpos_;
  MonitorElement* meOccupancyZneg_;

};

//
// constants, enums and typedefs
//

struct MTDHit {
  float energy;
  float time;
  float x;
  float y;
  float z;
};

//
// static data member definitions
//

//
// constructors and destructor
//
EtlSimHitsValidation::EtlSimHitsValidation(const edm::ParameterSet& iConfig):
  geom_(nullptr),
  folder_(iConfig.getParameter<std::string>("folder")),
  g4InfoLabel_(iConfig.getParameter<std::string>("moduleLabelG4")),
  etlHitsCollection_(iConfig.getParameter<std::string>("etlSimHitsCollection")),
  hitMinEnergy_( iConfig.getParameter<double>("hitMinimumEnergy") ),
  eventCount_(0) {
  
  etlSimHitsToken_ = consumes <edm::PSimHitContainer> (edm::InputTag(std::string(g4InfoLabel_),
								     std::string(etlHitsCollection_)));

}


EtlSimHitsValidation::~EtlSimHitsValidation() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called for each event  ------------
void EtlSimHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;

  edm::LogInfo("EventInfo") << " Run = " << iEvent.id().run() << " Event = " << iEvent.id().event();

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  geom_ = geometryHandle.product();

  edm::Handle<edm::PSimHitContainer> etlSimHitsHandle;
  iEvent.getByToken(etlSimHitsToken_, etlSimHitsHandle);

  if( ! etlSimHitsHandle.isValid() ) return;

  eventCount_++;
  
  std::map<uint32_t, MTDHit> m_etlHits;

  // --- Loop over the BLT SIM hits
  for (auto const& simHit: *etlSimHitsHandle) {

    // --- Only hits compatible with the in-time bunch-crossing
    if ( simHit.tof() < 0 || simHit.tof() > 25. ) continue;

    DetId id = simHit.detUnitId();

    auto simHitIt = m_etlHits.emplace(id.rawId(),MTDHit()).first;

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

  for (auto const& hit: m_etlHits) {

    if ( (hit.second).energy < hitMinEnergy_ ) continue;

    // --- Get the SIM hit global position
    ETLDetId detId(hit.first); 

    DetId geoId = detId.geographicalId();
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if( thedet == nullptr )
      throw cms::Exception("EtlSimHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId()
						   << " (" << detId.rawId()<< ") is invalid!" << std::dec
						   << std::endl;

    Local3DPoint local_point(0.1*(hit.second).x,0.1*(hit.second).y,0.1*(hit.second).z);
    const auto& global_point = thedet->toGlobal(local_point);

    // --- Fill the histograms
    if ( detId.zside()>0. ){

      meHitEnergyZpos_->Fill((hit.second).energy);
      meHitTimeZpos_->Fill((hit.second).time);
      meOccupancyZpos_->Fill(global_point.x(),global_point.y());

    }
    else {

      meHitEnergyZneg_->Fill((hit.second).energy);
      meHitTimeZneg_->Fill((hit.second).time);
      meOccupancyZneg_->Fill(global_point.x(),global_point.y());

    }

  } // hit loop

}

void EtlSimHitsValidation::bookHistograms(DQMStore::IBooker & ibook,
                               edm::Run const& run,
                               edm::EventSetup const & iSetup) {

  ibook.setCurrentFolder(folder_);

  meHitEnergyZpos_ = ibook.book1D("EtlHitEnergyZpos", "ETL SIM hits energy (+Z);E_{SIM} [MeV]", 200, 0., 2.);
  meHitEnergyZneg_ = ibook.book1D("EtlHitEnergyZneg", "ETL SIM hits energy (-Z);E_{SIM} [MeV]", 200, 0., 2.);
  meHitTimeZpos_   = ibook.book1D("EtlHitTimeZpos", "ETL SIM hits ToA (+Z);ToA_{SIM} [ns]", 250, 0., 25.);
  meHitTimeZneg_   = ibook.book1D("EtlHitTimeZneg", "ETL SIM hits ToA (-Z);ToA_{SIM} [ns]", 250, 0., 25.);

  meOccupancyZpos_ = ibook.book2D("EtlOccupancyZpos", "ETL SIM hits occupancy (+Z);x_{SIM} [cm];y_{SIM} [cm]",
				  135, -135., 135.,  135, -135., 135.);
  meOccupancyZneg_ = ibook.book2D("EtlOccupancyZneg", "ETL SIM hits occupancy (-Z);x_{SIM} [cm];y_{SIM} [cm]",
				  135, -135., 135.,  135, -135., 135.);

}



// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void EtlSimHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no
  // validation
  // Please change this to state exactly what you do use, even if it is no
  // parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "MTD/ETL/SimHits");
  desc.add<std::string>("moduleLabelG4","g4SimHits");
  desc.add<std::string>("etlSimHitsCollection","FastTimerHitsEndcap");
  desc.add<double>("hitMinimumEnergy",0.1); // [MeV]
  descriptions.add("etlSimHits", desc);

}

//define this as a plug-in
DEFINE_FWK_MODULE(EtlSimHitsValidation);
