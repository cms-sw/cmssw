// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

// Geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEbackDigitizer.h"

//ROOT headers
#include <TProfile2D.h>
#include <TH2F.h>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//STL headers
#include <vector>
#include <sstream>
#include <string>

//
// class declaration
//

class HGCHEbackSignalScalerAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
	public:
		explicit HGCHEbackSignalScalerAnalyzer(const edm::ParameterSet&);
		~HGCHEbackSignalScalerAnalyzer() override;

	private:
		void beginJob() override {}
		void analyze(const edm::Event&, const edm::EventSetup&) override;
		void endJob() override {}

    void createRadiusMap();
    void createZVector();

		// ----------member data ---------------------------
		edm::Service<TFileService> fs;

    std::string doseMap_;
    std::map<int, std::map<int, float>> layerRadiusMap_;

    const HGCalGeometry* gHGCal_;
    const HGCalDDDConstants* hgcCons_;

    int iLayerMin_ = 9;
    int iLayerMax_ = 24;
    int iRadiusMin_ = 1;
    int iRadiusMax_ = 49;

    float radiusMin_ = 70; //cm
    float radiusMax_ = 280; //cm
    int radiusBins_ = 8400;
    //old geometry from TDR
    //double xBins_[18] = {380.0, 395.1, 399.8, 404.7, 409.6, 417.5, 426.2, 434.9, 443.6, 452.3, 461.0, 469.7, 478.4, 487.1, 495.8, 504.5, 510.0, 515.5};
    double xBins_[17];
    int nxBins_ = sizeof(xBins_)/sizeof(*xBins_) - 1;

    bool verbose_ = false;
};

//
// constructors and destructor
//
HGCHEbackSignalScalerAnalyzer::HGCHEbackSignalScalerAnalyzer(const edm::ParameterSet& iConfig) :
	doseMap_(iConfig.getParameter<std::string>("doseMap"))
{
	usesResource("TFileService");
  fs->file().cd();
}


HGCHEbackSignalScalerAnalyzer::~HGCHEbackSignalScalerAnalyzer()
{
}


//
// member functions
//

// ------------ method called on each new Event  ------------
void
HGCHEbackSignalScalerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //get geometry
  edm::ESHandle<HGCalGeometry> geomhandle;
  iSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive", geomhandle);
  if(!geomhandle.isValid())
  {
    edm::LogError("HGCHEbackSignalScalerAnalyzer") << "Cannot get valid HGCalGeometry Object";
    return;
  }
  gHGCal_ = geomhandle.product();
  const std::vector<DetId>& detIdVec = gHGCal_->getValidDetIds();

  //get ddd constants
  edm::ESHandle<HGCalDDDConstants> dddhandle;
  iSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive",dddhandle);
  if (!dddhandle.isValid()) {
    edm::LogError("HGCHEbackSignalScalerAnalyzer") << "Cannot initiate HGCalDDDConstants";
    return;
  }
  hgcCons_ = dddhandle.product();


  //setup histos
  createRadiusMap();
  createZVector();
  TProfile2D* doseMap = fs->make<TProfile2D>("doseMap","doseMap", nxBins_, xBins_, radiusBins_, radiusMin_, radiusMax_);
  TProfile2D* scaleByDoseMap = fs->make<TProfile2D>("scaleByDoseMap","scaleByDoseMap", nxBins_, xBins_, radiusBins_, radiusMin_, radiusMax_);
  TProfile2D* scaleByAreaMap = fs->make<TProfile2D>("scaleByAreaMap","scaleByAreaMap", nxBins_, xBins_, radiusBins_, radiusMin_, radiusMax_);
  TProfile2D* scaleByAll = fs->make<TProfile2D>("scaleByAll","scaleByAll", nxBins_, xBins_, radiusBins_, radiusMin_, radiusMax_);

  //instantiate scaler
  HGCHEbackSignalScaler scal(gHGCal_, doseMap_);

  //loop over valid detId from the HGCHEback
  std::cout << "Total number of DetIDs: " << detIdVec.size() << std::endl;
  for(std::vector<DetId>::const_iterator myId = detIdVec.begin(); myId != detIdVec.end(); ++myId)
  {
    HGCScintillatorDetId scId(myId->rawId());
    double dose = scal.getDoseValue(scId);
    float scaleFactorByDose = scal.scaleByDose(scId);
    float scaleFactorByArea = scal.scaleByArea(scId);

    int ilayer = scId.layer();
    int iradius = scId.iradiusAbs();
    std::pair<double,double> cellSize = hgcCons_->cellSizeTrap(scId.type(), scId.iradiusAbs());
    float inradius = cellSize.first;

    GlobalPoint global = gHGCal_->getPosition(scId);
    float zpos = std::abs(global.z());

    int bin = doseMap->GetYaxis()->FindBin(inradius);
    while(scaleByDoseMap->GetYaxis()->GetBinLowEdge(bin) < layerRadiusMap_[ilayer][iradius+1])
    {
      if(verbose_)
        std::cout << "rIN = " << layerRadiusMap_[ilayer][iradius]
                  << " rIN+1 = " << layerRadiusMap_[ilayer][iradius+1]
                  << " inradius = " << inradius
                  << " type = " << scId.type()
                  << " ilayer = " << scId.layer() << std::endl;

      doseMap->Fill(zpos, scaleByDoseMap->GetYaxis()->GetBinCenter(bin), dose);
      scaleByDoseMap->Fill(zpos, scaleByDoseMap->GetYaxis()->GetBinCenter(bin), scaleFactorByDose);
      scaleByAreaMap->Fill(zpos, scaleByDoseMap->GetYaxis()->GetBinCenter(bin), scaleFactorByArea);
      scaleByAll->Fill(zpos, scaleByDoseMap->GetYaxis()->GetBinCenter(bin), scaleFactorByArea * scaleFactorByDose);
      ++bin;
    }

  }
}


void HGCHEbackSignalScalerAnalyzer::createRadiusMap()
{
  for(int layer=iLayerMin_; layer<iLayerMax_+1; ++layer)
    for(int radius=iRadiusMin_; radius<iRadiusMax_+1; ++radius)
    {
      HGCScintillatorDetId scId(hgcCons_->getTypeTrap(layer), layer, radius, 1);
      layerRadiusMap_[layer][radius] = (hgcCons_->cellSizeTrap(scId.type(), scId.iradiusAbs())).first; //internal radius
    }
}

void HGCHEbackSignalScalerAnalyzer::createZVector()
{
  int iBin=0;
  for(int layer=iLayerMin_; layer<iLayerMax_+1; ++layer)
  {
    HGCScintillatorDetId scId(hgcCons_->getTypeTrap(layer), layer, 20, 1);
    GlobalPoint global = gHGCal_->getPosition(scId);
    xBins_[iBin] = global.z();
    ++iBin;
  }
  //guess the last bin
  xBins_[iBin] = xBins_[iBin-1] + (xBins_[iBin-1]-xBins_[iBin-2]);
}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCHEbackSignalScalerAnalyzer);
