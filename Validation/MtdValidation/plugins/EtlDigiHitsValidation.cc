// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      EtlDigiHitsValidation
//
/**\class EtlDigiHitsValidation EtlDigiHitsValidation.cc Validation/MtdValidation/plugins/EtlDigiHitsValidation.cc

 Description: ETL DIGI hits validation

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

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"

#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"


class EtlDigiHitsValidation : public DQMEDAnalyzer {

public:
  explicit EtlDigiHitsValidation(const edm::ParameterSet&);
  ~EtlDigiHitsValidation() override;

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
  const std::string etlDigisCollection_;

  int eventCount_;

  edm::EDGetTokenT<ETLDigiCollection> etlDigiHitsToken_;

  // --- histograms declaration

  MonitorElement* meHitCharge_[2];
  MonitorElement* meHitTime_[2];

  MonitorElement* meOccupancy_[2];

};


// ------------ constructor and destructor --------------
EtlDigiHitsValidation::EtlDigiHitsValidation(const edm::ParameterSet& iConfig):
  geom_(nullptr),
  folder_(iConfig.getParameter<std::string>("folder")),
  infoLabel_(iConfig.getParameter<std::string>("moduleLabel")),
  etlDigisCollection_(iConfig.getParameter<std::string>("etlDigiHitsCollection")),
  eventCount_(0) {

  etlDigiHitsToken_ = consumes <ETLDigiCollection> (edm::InputTag(std::string(infoLabel_),
								  std::string(etlDigisCollection_)));

}

EtlDigiHitsValidation::~EtlDigiHitsValidation() {
}


// ------------ method called for each event  ------------
void EtlDigiHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;

  edm::LogInfo("EventInfo") << " Run = " << iEvent.id().run() << " Event = " << iEvent.id().event();

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  geom_ = geometryHandle.product();

  edm::Handle<ETLDigiCollection> etlDigiHitsHandle;
  iEvent.getByToken(etlDigiHitsToken_, etlDigiHitsHandle);

  if( ! etlDigiHitsHandle.isValid() ) {
    edm::LogWarning("DataNotFound") << "No ETL DIGI hits found";
    return;
  }

  eventCount_++;
  
  // --- Loop over the ELT DIGI hits
  for (const auto& dataFrame: *etlDigiHitsHandle) {

    // --- Get the on-time sample
    int isample = 2;

    const auto& sample = dataFrame.sample(isample);

    ETLDetId detId = dataFrame.id();

    DetId geoId = detId.geographicalId();

    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if( thedet == nullptr )
      throw cms::Exception("EtlDigiHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId()
						    << " (" << detId.rawId()<< ") is invalid!" << std::dec
						    << std::endl;
    
    Local3DPoint local_point(0.,0.,0.);
    const auto& global_point = thedet->toGlobal(local_point);

    // --- Fill the histograms

    int idet = (detId.zside()+1)/2;

    meHitCharge_[idet]->Fill(sample.data());
    meHitTime_[idet]->Fill(sample.toa());
    meOccupancy_[idet]->Fill(global_point.x(),global_point.y());

  } // dataFrame loop

}


// ------------ method for histogram booking ------------
void EtlDigiHitsValidation::bookHistograms(DQMStore::IBooker & ibook,
                               edm::Run const& run,
                               edm::EventSetup const & iSetup) {

  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meHitCharge_[0] = ibook.book1D("EtlHitChargeZneg", "ETL DIGI hits charge (-Z);amplitude [ADC counts]",
				 256, 0., 256.);
  meHitCharge_[1] = ibook.book1D("EtlHitChargeZpos", "ETL DIGI hits charge (+Z);amplitude [ADC counts]",
				 256, 0., 256.);

  meHitTime_[0]   = ibook.book1D("EtlHitTimeZneg", "ETL DIGI hits ToA (-Z);ToA [TDC counts]", 
				 1000, 0., 2000.);
  meHitTime_[1]   = ibook.book1D("EtlHitTimeZpos", "ETL DIGI hits ToA (+Z);ToA [TDC counts]", 
				 1000, 0., 2000.);

  meOccupancy_[0] = ibook.book2D("EtlOccupancyZneg","ETL DIGI hits occupancy (-Z);x [cm];y [cm]",
				 135, -135., 135.,  135, -135., 135.);
  meOccupancy_[1] = ibook.book2D("EtlOccupancyZpos","ETL DIGI hits occupancy (+Z);x [cm];y [cm]",
				 135, -135., 135.,  135, -135., 135.);

}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void EtlDigiHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ETL/DigiHits");
  desc.add<std::string>("moduleLabel","mix");
  desc.add<std::string>("etlDigiHitsCollection","FTLEndcap");

  descriptions.add("etlDigiHits", desc);

}

DEFINE_FWK_MODULE(EtlDigiHitsValidation);
