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

		// ----------member data ---------------------------
		edm::Service<TFileService> fs;
    std::string doseMap_;
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
  edm::ESHandle<HGCalGeometry> handle;
  iSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive", handle);
  if(!handle.isValid())
  {
    edm::LogError("HGCHEbackSignalScalerAnalyzer") << "Cannot get valid HGCalGeometry Object";
    return;
  }

  const HGCalGeometry* gHGCal = handle.product();
  const std::vector<DetId>& detIdVec = gHGCal->getValidDetIds();

  //setup scaler and histo
  int layerMin = 9;
  int layerMax = 24;
  int iradiusMin = 1;
  int iradiusMax = 48;

  TProfile2D* doseMap = fs->make<TProfile2D>("doseMap","doseMap", layerMax-layerMin+1, layerMin, layerMax+1, iradiusMax-iiradiusMin+1, iradiusMin, iradiusMax+1);
  TProfile2D* scaleByDoseMap = fs->make<TProfile2D>("scaleByDoseMap","scaleByDoseMap", layerMax-layerMin+1, layerMin, layerMax+1, iradiusMax-iradiusMin+1, iradiusMin, iradiusMax+1);
  TProfile2D* scaleByAreaMap = fs->make<TProfile2D>("scaleByAreaMap","scaleByAreaMap", layerMax-layerMin+1, layerMin, layerMax+1, iradiusMax-iradiusMin+1, iradiusMin, iradiusMax+1);

  HGCHEbackSignalScaler scal(gHGCal, doseMap_);

  //loop over valid detId from the HGCHEback
  std::cout << "Total number of DetIDs: " << detIdVec.size() << std::endl;
  for(std::vector<DetId>::const_iterator myId = detIdVec.begin(); myId != detIdVec.end(); ++myId)
  {
    HGCScintillatorDetId scId(myId->rawId());
    float scaleFactorByDose = scal.scaleByDose(scId);
    double dose = scal.getDoseValue(scId);
    float scaleFactorByArea = scal.scaleByArea(scId);

    GlobalPoint global = gHGCal->getPosition(scId);
    //float radius = sqrt( std::pow(global.x(), 2) + std::pow(global.y(), 2));


    doseMap->Fill(scId.layer(), scId.iradiusAbs(), dose);
    scaleByDoseMap->Fill(scId.layer(), scId.iradiusAbs(), scaleFactorByDose);
    scaleByAreaMap->Fill(scId.layer(), scId.iradiusAbs(), scaleFactorByArea);

  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCHEbackSignalScalerAnalyzer);
