#include <memory>
#include <iostream>

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimDataFormats/ValidationFormats/interface/PHGCalValidInfo.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include <TH2.h>

//#define EDM_ML_DEBUG

class HGCGeometryCheck : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:

  explicit HGCGeometryCheck( const edm::ParameterSet& );
  ~HGCGeometryCheck();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  virtual void beginJob() override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

private:
  edm::EDGetTokenT<PHGCalValidInfo> g4Token_;
  std::vector<std::string>          geometrySource_;

  //HGCal geometry scheme
  std::vector<const HGCalDDDConstants*> hgcGeometry_;
  const HcalDDDSimConstants *hcons_;

  //histogram related stuff
  TH2F           *heedzVsZ,    *hefdzVsZ,    *hebdzVsZ;
  TH2F           *heezVsLayer, *hefzVsLayer, *hebzVsLayer;
  TH2F           *heerVsLayer, *hefrVsLayer, *hebrVsLayer;

};

HGCGeometryCheck::HGCGeometryCheck(const edm::ParameterSet &cfg) : hcons_(0) {

  usesResource("TFileService");

  g4Token_ = consumes<PHGCalValidInfo>(cfg.getParameter<edm::InputTag>("g4Source"));
  geometrySource_ = cfg.getUntrackedParameter< std::vector<std::string> >("geometrySource");
#ifdef EDM_ML_DEBUG
  std::cout << "HGCGeometryCheck:: use information from "
	    << cfg.getParameter<edm::InputTag>("g4Source") << " and "
	    << geometrySource_.size() << " geometry records:";
  for (unsigned int k=0; k<geometrySource_.size(); ++k)
    std::cout << " " << geometrySource_[k];
  std::cout << std::endl;
#endif
}

HGCGeometryCheck::~HGCGeometryCheck() { }

void HGCGeometryCheck::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HGCGeometryCheck::beginJob() {

  //initiating fileservice
  edm::Service<TFileService> fs;

  //initiating histograms
  heedzVsZ     = fs->make<TH2F>("heedzVsZ","", 800,315,355,100,-1,1);
  hefdzVsZ     = fs->make<TH2F>("hefdzVsZ","",1200,350,410,100,-1,1);
  hebdzVsZ     = fs->make<TH2F>("hebdzVsZ","", 320,400,560,100,-5,5);

  heezVsLayer = fs->make<TH2F>("heezVsLayer","",100,0,100, 800,315,355);
  hefzVsLayer = fs->make<TH2F>("hefzVsLayer","", 40,0, 40,1200,350,410);
  hebzVsLayer = fs->make<TH2F>("hebzVsLayer","", 50,0, 25, 320,400,560);

  heerVsLayer = fs->make<TH2F>("heerVsLayer","",100,0,100,600,0,300);
  hefrVsLayer = fs->make<TH2F>("hefrVsLayer","", 40,0, 40,600,0,300);
  hebrVsLayer = fs->make<TH2F>("hebrVsLayer","", 50,0, 25,600,0,300);
}

void HGCGeometryCheck::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {

  //initiating hgc geometry
  for (size_t i=0; i<geometrySource_.size(); i++) {
    if (geometrySource_[i].find("Hcal") != std::string::npos) {
      edm::ESHandle<HcalDDDSimConstants> pHRNDC;
      iSetup.get<HcalSimNumberingRecord>().get(pHRNDC);
      if (pHRNDC.isValid()) {
	hcons_ = &(*pHRNDC);
	hgcGeometry_.push_back(0);
      } else {
	edm::LogWarning("HGCalValid") << "Cannot initiate HcalGeometry for "
				      << geometrySource_[i] << std::endl;
      }
    } else {
      edm::ESHandle<HGCalDDDConstants> hgcGeom;
      iSetup.get<IdealGeometryRecord>().get(geometrySource_[i],hgcGeom);
      if (hgcGeom.isValid()) {
	hgcGeometry_.push_back(hgcGeom.product());
      } else {
	edm::LogWarning("HGCalValid") << "Cannot initiate HGCalGeometry for "
				      << geometrySource_[i] << std::endl;
      }
    }
  }
}

void HGCGeometryCheck::analyze(const edm::Event &iEvent, 
			       const edm::EventSetup &iSetup) {

#ifdef EDM_ML_DEBUG
  std::cout << "HGCGeometryCheck::Run " << iEvent.id().run() << " Event " 
	    << iEvent.id().event() << " Luminosity " 
	    << iEvent.luminosityBlock() << " Bunch " 
	    << iEvent.bunchCrossing() << std::endl;

#endif
  //Accessing G4 information
  edm::Handle<PHGCalValidInfo> infoLayer;
  iEvent.getByToken(g4Token_,infoLayer);

  if (infoLayer.isValid()) {
    //step vertex information
    std::vector<float> hitVtxX = infoLayer->hitvtxX();
    std::vector<float> hitVtxY = infoLayer->hitvtxY();
    std::vector<float> hitVtxZ = infoLayer->hitvtxZ();
    std::vector<unsigned int> hitDet = infoLayer->hitDets();
    std::vector<unsigned int> hitIdx = infoLayer->hitIndex();
    
    //loop over all hits
    for (unsigned int i=0; i<hitVtxZ.size(); i++) {

      double xx = hitVtxX.at(i)/10;
      double yy = hitVtxY.at(i)/10;
      double zz = hitVtxZ.at(i)/10;
      double rr = sqrt(xx*xx+yy*yy);
      if (hitDet.at(i) == (unsigned int)(DetId::Forward)) {
	int subdet, zside, layer, wafer, celltype, cell;
	HGCalTestNumbering::unpackHexagonIndex(hitIdx.at(i), subdet, zside, layer, wafer, celltype, cell);	
	if (subdet==(int)(HGCEE)) {
	  double zp = hgcGeometry_[0]->waferZ(layer,false); //cm 
	  if (zside < 0) zp = -zp;
#ifdef EDM_ML_DEBUG
	  std::cout << "Info[" << i << "] Detector Information " << hitDet[i]
		    << ":" << subdet << ":" << zside << ":" << layer << ":"
		    << wafer << ":" << celltype << ":" << cell << " Z "
		    << zp << ":" << zz << " R " << rr << std::endl;
#endif
	  heedzVsZ->Fill(zp, (zz-zp));
	  heezVsLayer->Fill(layer,zz);
	  heerVsLayer->Fill(layer,rr);
	} else if (subdet==(int)(HGCHEF)) {
	  double zp = hgcGeometry_[1]->waferZ(layer,false); //cm 
	  if (zside < 0) zp = -zp;
#ifdef EDM_ML_DEBUG
	  std::cout << "Info[" << i << "] Detector Information " << hitDet[i]
		    << ":" << subdet << ":" << zside << ":" << layer << ":"
		    << wafer << ":" << celltype << ":" << cell << " z "
		    << zp << ":" << zz << " R " << rr << std::endl;
#endif
	  hefdzVsZ->Fill(zp, (zz-zp));
	  hefzVsLayer->Fill(layer,zz);
	  hefrVsLayer->Fill(layer,rr);
	}
	
      } else if (hitDet.at(i) == (unsigned int)(DetId::Hcal)) {

	int subdet, zside, depth, eta, phi, lay;
	HcalTestNumbering::unpackHcalIndex(hitIdx.at(i), subdet, zside, depth, eta, phi, lay);
	HcalCellType::HcalCell cell = hcons_->cell(subdet, zside, lay, eta, phi);
	double zp = cell.rz/10; //mm --> cm
	if (zside == 0) zp = -zp;
#ifdef EDM_ML_DEBUG
	std::cout << "Info[" << i << "] Detector Information " << hitDet[i]
		  << ":" << subdet << ":" << zside << ":" << depth << ":"
		  << eta << ":" << phi << ":" << lay << " z "  << zp << ":"
		  << zz << " R " << rr << std::endl;
#endif
	hebdzVsZ->Fill(zp, (zz-zp));
	hebzVsLayer->Fill(lay,zz);
	hebrVsLayer->Fill(lay,rr);
      }

    }//end G4 hits
    
  } else {
    edm::LogWarning("HGCalValid") << "No PHGCalInfo " << std::endl;
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCGeometryCheck);






