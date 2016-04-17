
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/ValidationFormats/interface/PHGCalValidInfo.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include "PhysicsTools/HepMCCandAlgos/interface/GenParticlesHelper.h"

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TSystem.h>
#include <TFile.h>
#include <TProfile.h>
#include <memory>
#include <iostream>

#define SEP "\t"

class HGCGeometryValidation : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:

  explicit HGCGeometryValidation( const edm::ParameterSet& );
  ~HGCGeometryValidation();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void beginJob();
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  edm::EDGetTokenT<PHGCalValidInfo> g4Token_;
  std::vector<std::string> geometrySource_;

  //HGCal geometry scheme
  std::vector<const HGCalDDDConstants*> hgcGeometry_;
  const HcalDDDSimConstants *hcons_;

  //histogram related stuff
  TH2F *heedzVsZ, *heedyVsY, *heedxVsX;
  TH2F *hefdzVsZ, *hefdyVsY, *hefdxVsX;
  TH2F *hebdzVsZ, *hebdyVsY, *hebdxVsX;
  TH2F *heedzVsLayer, *hefdzVsLayer, *hebdzVsLayer, *heedyVsLayer, *hefdyVsLayer, *hebdyVsLayer, *heedxVsLayer, *hefdxVsLayer, *hebdxVsLayer;
  TH2F *heeXG4VsId, *hefXG4VsId, *hebXG4VsId, *heeYG4VsId, *hefYG4VsId, *hebYG4VsId, *heeZG4VsId, *hefZG4VsId, *hebZG4VsId;
  TH2F *hebLayerVsEnStep, *hefLayerVsEnStep, *heeLayerVsEnStep;

	
  TH1F *heeTotEdepStep, *hefTotEdepStep, *hebTotEdepStep;
  TH1F *heedX, *heedY, *heedZ;
  TH1F *hefdX, *hefdY, *hefdZ;
  TH1F *hebdX, *hebdY, *hebdZ;
};


//
HGCGeometryValidation::HGCGeometryValidation(const edm::ParameterSet &cfg) : hcons_(0) {

  usesResource("TFileService");

  g4Token_ = consumes<PHGCalValidInfo>(cfg.getParameter<edm::InputTag>("g4Source"));
  geometrySource_ = cfg.getUntrackedParameter< std::vector<std::string> >("geometrySource");
}

//
HGCGeometryValidation::~HGCGeometryValidation() {
}

void HGCGeometryValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//
void HGCGeometryValidation::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {

  //initiating hgcnumbering
  for (size_t i=0; i<geometrySource_.size(); i++) {
                
    if (geometrySource_[i].find("Hcal") != std::string::npos) {
      edm::ESHandle<HcalDDDSimConstants> pHRNDC;
      iSetup.get<HcalSimNumberingRecord>().get(pHRNDC);
      if (pHRNDC.isValid()) {
	hcons_ = &(*pHRNDC);
	hgcGeometry_.push_back(0);
      } else {
	edm::LogWarning("HGCalValid") << "Cannot initiate HGCalGeometry for "
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


//
void HGCGeometryValidation::beginJob() {

  //initiating fileservice
  edm::Service<TFileService> fs;

  //initiating histograms
  heeTotEdepStep = fs->make<TH1F>("heeTotEdepStep","",100,0,100);
  hefTotEdepStep = fs->make<TH1F>("hefTotEdepStep","",100,0,100); 
  hebTotEdepStep = fs->make<TH1F>("hebTotEdepStep","",100,0,100);

  hebLayerVsEnStep = fs->make<TH2F>("hebLayerVsEnStep","",25,0,25,100,0,0.01);
  hefLayerVsEnStep = fs->make<TH2F>("hefLayerVsEnStep","",36,0,36,100,0,0.01);
  heeLayerVsEnStep = fs->make<TH2F>("heeLayerVsEnStep","",84,0,84,100,0,0.01);

  heeXG4VsId = fs->make<TH2F>("heeXG4VsId","",600,-300,300,600,-300,300);
  heeYG4VsId = fs->make<TH2F>("heeYG4VsId","",600,-300,300,600,-300,300);
  heeZG4VsId = fs->make<TH2F>("heeZG4VsId","",200,310,360,200,310,360);

  hefXG4VsId = fs->make<TH2F>("hefXG4VsId","",600,-300,300,600,-300,300);
  hefYG4VsId = fs->make<TH2F>("hefYG4VsId","",600,-300,300,600,-300,300);
  hefZG4VsId = fs->make<TH2F>("hefZG4VsId","",320,340,500,320,340,500);

  hebXG4VsId = fs->make<TH2F>("hebXG4VsId","",600,-300,300,600,-300,300);
  hebYG4VsId = fs->make<TH2F>("hebYG4VsId","",600,-300,300,600,-300,300);
  hebZG4VsId = fs->make<TH2F>("hebZG4VsId","",220,400,620,220,400,620);

  heedzVsZ = fs->make<TH2F>("heedzVsZ","",200,310,360,100,-1,1);
  heedyVsY = fs->make<TH2F>("heedyVsY","",400,-200,200,100,-1,1);
  heedxVsX = fs->make<TH2F>("heedxVsX","",400,-200,200,100,-1,1);

  hefdzVsZ = fs->make<TH2F>("hefdzVsZ","",320,340,500,100,-1,1);
  hefdyVsY = fs->make<TH2F>("hefdyVsY","",400,-200,200,100,-1,1);
  hefdxVsX = fs->make<TH2F>("hefdxVsX","",400,-200,200,100,-1,1);

  hebdzVsZ = fs->make<TH2F>("hebdzVsZ","",220,400,620,100,-5,5);
  hebdyVsY = fs->make<TH2F>("hebdyVsY","",400,-200,200,100,-5,5);
  hebdxVsX = fs->make<TH2F>("hebdxVsX","",400,-200,200,100,-5,5);

  heedzVsLayer = fs->make<TH2F>("heedzVsLayer","",100,0,100,100,-1,1);
  hefdzVsLayer = fs->make<TH2F>("hefdzVsLayer","",40,0,40,100,-1,1);
  hebdzVsLayer = fs->make<TH2F>("hebdzVsLayer","",50,0,25,100,-5,5);

  heedyVsLayer = fs->make<TH2F>("heedyVsLayer","",100,0,100,100,-1,1);
  hefdyVsLayer = fs->make<TH2F>("hefdyVsLayer","",40,0,40,100,-1,1);
  hebdyVsLayer = fs->make<TH2F>("hebdyVsLayer","",50,0,25,100,-5,5);

  heedxVsLayer = fs->make<TH2F>("heedxVsLayer","",100,0,100,100,-1,1);
  hefdxVsLayer = fs->make<TH2F>("hefdxVsLayer","",40,0,40,500,-1,1);
  hebdxVsLayer = fs->make<TH2F>("hebdxVsLayer","",50,0,25,500,-5,5.0);

  heedX = fs->make<TH1F>("heedX","",100,-1,1); 
  heedY = fs->make<TH1F>("heedY","",100,-1,1);
  heedZ = fs->make<TH1F>("heedZ","",100,-1,1);

  hefdX = fs->make<TH1F>("hefdX","",100,-1,1); 
  hefdY = fs->make<TH1F>("hefdY","",100,-1,1);
  hefdZ = fs->make<TH1F>("hefdZ","",100,-1,1);

  hebdX = fs->make<TH1F>("hebdX","",100,-1,1); 
  hebdY = fs->make<TH1F>("hebdY","",100,-1,1);
  hebdZ = fs->make<TH1F>("hebdZ","",100,-1,1);

}


//
void HGCGeometryValidation::analyze( const edm::Event &iEvent, const edm::EventSetup &iSetup) {

  //Accessing G4 information
  try{
    edm::Handle<PHGCalValidInfo> infoLayer;
    iEvent.getByToken(g4Token_,infoLayer);
	
    //step vertex information
    std::vector<float> hitVtxX = infoLayer->hitvtxX();
    std::vector<float> hitVtxY = infoLayer->hitvtxY();
    std::vector<float> hitVtxZ = infoLayer->hitvtxZ();
    std::vector<unsigned int> hitDet = infoLayer->hitDets();
    std::vector<unsigned int> hitIdx = infoLayer->hitIndex();

    //energy information
    std::vector<float> edepLayerEE = infoLayer->eehgcEdep();
    std::vector<float> edepLayerHE = infoLayer->hefhgcEdep();
    std::vector<float> edepLayerHB = infoLayer->hebhgcEdep();
    
    unsigned int i;		
    for(i=0; i<edepLayerEE.size(); i++) {
      heeLayerVsEnStep->Fill(i,edepLayerEE.at(i));
    }
	
    for(i=0; i<edepLayerHE.size(); i++)	{
      hefLayerVsEnStep->Fill(i,edepLayerHE.at(i));
    }

    for(i=0; i<edepLayerHB.size(); i++) {
      hebLayerVsEnStep->Fill(i,edepLayerHB.at(i));
    }

    //fill total energy deposited
    heeTotEdepStep->Fill((double)infoLayer->eeTotEdep());
    hefTotEdepStep->Fill((double)infoLayer->hefTotEdep()); 
    hebTotEdepStep->Fill((double)infoLayer->hebTotEdep());
    
    //loop over all hits
    for(unsigned int i=0; i<hitVtxX.size(); i++) {

      if (hitDet.at(i) == (unsigned int)(DetId::Forward)) {
	int subdet, zside, layer, wafer, celltype, cell;
	HGCalTestNumbering::unpackHexagonIndex(hitIdx.at(i), subdet, zside, layer, wafer, celltype, cell);	

	std::pair<float, float> xy;
	std::pair<int,float> layerIdx;
	double zp, xx, yx;

	if (subdet==(int)(HGCEE)) {
	  xy = hgcGeometry_[0]->locateCell(cell,layer,wafer,false); //mm
	  zp = hgcGeometry_[0]->waferZ(layer,false); //cm 
	  xx = (zp<0) ? -xy.first/10 : xy.first/10; //mm
	  yx = xy.second/10; //mm
	  hitVtxX.at(i) = hitVtxX.at(i)/10;
	  hitVtxY.at(i) = hitVtxY.at(i)/10;
	  hitVtxZ.at(i) = hitVtxZ.at(i)/10;
	  
	  heedzVsZ->Fill(zp, (hitVtxZ.at(i)-zp));
	  heedyVsY->Fill(yx, (hitVtxY.at(i)-yx));
	  heedxVsX->Fill(xx, (hitVtxX.at(i)-xx));
	  
	  heeXG4VsId->Fill(hitVtxX.at(i),xx);
	  heeYG4VsId->Fill(hitVtxY.at(i),yx);
	  heeZG4VsId->Fill(hitVtxZ.at(i),zp);

	  heedzVsLayer->Fill(layer,(hitVtxZ.at(i)-zp));
	  heedyVsLayer->Fill(layer,(hitVtxY.at(i)-yx));
	  heedxVsLayer->Fill(layer,(hitVtxX.at(i)-xx));
	  
	  heedX->Fill((hitVtxX.at(i)-xx));
	  heedZ->Fill((hitVtxZ.at(i)-zp));
	  heedY->Fill((hitVtxY.at(i)-yx));

	} else if (subdet==(int)(HGCHEF)) {

	  xy = hgcGeometry_[1]->locateCell(cell,layer,wafer,false); //mm
	  zp = hgcGeometry_[1]->waferZ(layer,false); //cm 
	  xx = (zp<0) ? -xy.first/10 : xy.first/10; //mm
	  yx = xy.second/10; //mm
	  hitVtxX.at(i) = hitVtxX.at(i)/10;
	  hitVtxY.at(i) = hitVtxY.at(i)/10;
	  hitVtxZ.at(i) = hitVtxZ.at(i)/10;
	  
	  hefdzVsZ->Fill(zp, (hitVtxZ.at(i)-zp));
	  hefdyVsY->Fill(yx, (hitVtxY.at(i)-yx));
	  hefdxVsX->Fill(xx, (hitVtxX.at(i)-xx));
	  
	  hefXG4VsId->Fill(hitVtxX.at(i),xx);
	  hefYG4VsId->Fill(hitVtxY.at(i),yx);
	  hefZG4VsId->Fill(hitVtxZ.at(i),zp);
	  
	  hefdzVsLayer->Fill(layer,(hitVtxZ.at(i)-zp));
	  hefdyVsLayer->Fill(layer,(hitVtxY.at(i)-yx));
	  hefdxVsLayer->Fill(layer,(hitVtxX.at(i)-xx));
	  
	  hefdX->Fill((hitVtxX.at(i)-xx));
	  hefdZ->Fill((hitVtxZ.at(i)-zp));
	  hefdY->Fill((hitVtxY.at(i)-yx));
	
	}
	
      } else if (hitDet.at(i) == (unsigned int)(DetId::Hcal)) {

	int subdet, zside, depth, eta, phi, lay;
	HcalTestNumbering::unpackHcalIndex(hitIdx.at(i), subdet, zside, depth, eta, phi, lay);
	HcalCellType::HcalCell cell = hcons_->cell(subdet, zside, lay, eta, phi);
	
	double zp = cell.rz/10; //mm --> cm
	double rho = zp*TMath::Tan(2.0*TMath::ATan(TMath::Exp(-cell.eta)));
	double xp = rho * TMath::Cos(cell.phi); //cm
	double yp = rho * TMath::Sin(cell.phi); //cm

	hitVtxX.at(i) = hitVtxX.at(i)/10;
	hitVtxY.at(i) = hitVtxY.at(i)/10;
	hitVtxZ.at(i) = hitVtxZ.at(i)/10;

	hebdzVsZ->Fill(zp, (hitVtxZ.at(i)-zp));
	hebdyVsY->Fill(yp, (hitVtxY.at(i)-yp));
	hebdxVsX->Fill(xp, (hitVtxX.at(i)-xp));

	hebXG4VsId->Fill(hitVtxX.at(i),xp);
	hebYG4VsId->Fill(hitVtxY.at(i),yp);
	hebZG4VsId->Fill(hitVtxZ.at(i),zp);

	hebdzVsLayer->Fill(lay,(hitVtxZ.at(i)-zp));
	hebdyVsLayer->Fill(lay,(hitVtxY.at(i)-yp));
	hebdxVsLayer->Fill(lay,(hitVtxX.at(i)-xp));

	hebdX->Fill((hitVtxX.at(i)-xp));
	hebdZ->Fill((hitVtxZ.at(i)-zp));
	hebdY->Fill((hitVtxY.at(i)-yp));
      }

    }//end G4 hits
    
  } catch(...) {
    edm::LogWarning("HGCalValid") << "No PHGCalInfo " << std::endl;
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCGeometryValidation);






