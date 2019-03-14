// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      EtlRecHitsValidation
//
/**\class EtlRecHitsValidation EtlRecHitsValidation.cc Validation/MtdValidation/plugins/EtlRecHitsValidation.cc

 Description: ETL RECO hits validation

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

#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"


class EtlRecHitsValidation : public DQMEDAnalyzer {

public:
  explicit EtlRecHitsValidation(const edm::ParameterSet&);
  ~EtlRecHitsValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  void bookHistograms(DQMStore::IBooker &,
		      edm::Run const&,
		      edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const MTDGeometry* geom_;

  const std::string folder_;
  const std::string infoLabel_;
  const std::string etlRecHitsCollection_;

  int eventCount_;

  edm::EDGetTokenT<FTLRecHitCollection> etlRecHitsToken_;

  // --- histograms declaration

  MonitorElement* meHitEnergy_[2];
  MonitorElement* meHitTime_[2];

  MonitorElement* meOccupancy_[2];

};


// ------------ constructor and destructor --------------
EtlRecHitsValidation::EtlRecHitsValidation(const edm::ParameterSet& iConfig):
  geom_(nullptr),
  folder_(iConfig.getParameter<std::string>("folder")),
  infoLabel_(iConfig.getParameter<std::string>("moduleLabel")),
  etlRecHitsCollection_(iConfig.getParameter<std::string>("etlRecHitsCollection")),
  eventCount_(0) {

  etlRecHitsToken_ = consumes <FTLRecHitCollection> (edm::InputTag(std::string(infoLabel_),
								   std::string(etlRecHitsCollection_)));

}

EtlRecHitsValidation::~EtlRecHitsValidation() {
}


// ------------ method called for each event  ------------
void EtlRecHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;

  edm::LogInfo("EventInfo") << " Run = " << iEvent.id().run() << " Event = " << iEvent.id().event();

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  geom_ = geometryHandle.product();

  edm::Handle<FTLRecHitCollection> etlRecHitsHandle;
  iEvent.getByToken(etlRecHitsToken_, etlRecHitsHandle);

  if( ! etlRecHitsHandle.isValid() ) {
    edm::LogWarning("DataNotFound") << "No ETL RecHits found";
    return;
  }

  eventCount_++;

  // --- Loop over the ELT RECO hits
  for (const auto& recHit: *etlRecHitsHandle) {

    ETLDetId detId = recHit.id();

    DetId geoId = detId.geographicalId();
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if( thedet == nullptr )
      throw cms::Exception("EtlRecHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId()
						   << " (" << detId.rawId()<< ") is invalid!" << std::dec
						   << std::endl;

    const auto& global_point = thedet->toGlobal(Local3DPoint(0.,0.,0.));

    // --- Fill the histograms

    int idet = (detId.zside()+1)/2;

    meHitEnergy_[idet]->Fill(recHit.energy());
    meHitTime_[idet]->Fill(recHit.time());

    meOccupancy_[idet]->Fill(global_point.x(),global_point.y());

  } // recHit loop

}


// ------------ method for histogram booking ------------
void EtlRecHitsValidation::bookHistograms(DQMStore::IBooker & ibook,
                               edm::Run const& run,
                               edm::EventSetup const & iSetup) {

  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meHitEnergy_[0] = ibook.book1D("EtlHitEnergyZneg", "ETL RECO hits energy (-Z);energy [MeV]", 150, 0., 3.);
  meHitEnergy_[1] = ibook.book1D("EtlHitEnergyZpos", "ETL RECO hits energy (+Z);energy [MeV]", 150, 0., 3.);

  meHitTime_[0] = ibook.book1D("EtlHitTimeZneg", "ETL RECO hits ToA (-Z);ToA [ns]", 250, 0., 25.);
  meHitTime_[1] = ibook.book1D("EtlHitTimeZpos", "ETL RECO hits ToA (+Z);ToA [ns]", 250, 0., 25.);

  meOccupancy_[0] = ibook.book2D("EtlOccupancyZneg","ETL RECO hits occupancy (-Z);x [cm];y [cm]",
				 59, -130., 130.,  59, -130., 130.);
  meOccupancy_[1] = ibook.book2D("EtlOccupancyZpos","ETL DIGI hits occupancy (+Z);x [cm];y [cm]",
				 59, -130., 130.,  59, -130., 130.);

}



// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void EtlRecHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ETL/RecHits");
  desc.add<std::string>("moduleLabel","mtdRecHits");
  desc.add<std::string>("etlRecHitsCollection","FTLEndcap");

  descriptions.add("etlRecHits", desc);

}

DEFINE_FWK_MODULE(EtlRecHitsValidation);
