/**_________________________________________________________________
   class:   BeamSpotWrite2DB.h
   package: RecoVertex/BeamSpotProducer

   author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
________________________________________________________________**/

// C++ standard
#include <string>
#include <fstream>

// CMS
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotWrite2DB.h"

// ROOT
#include "TFile.h"
#include "TTree.h"

class BeamSpotWrite2DB : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BeamSpotWrite2DB(const edm::ParameterSet&);
  ~BeamSpotWrite2DB() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  std::ifstream fasciiFile;
  std::string fasciiFileName;
};

BeamSpotWrite2DB::BeamSpotWrite2DB(const edm::ParameterSet& iConfig) {
  usesResource("PoolDBOutputService");
  fasciiFileName = iConfig.getUntrackedParameter<std::string>("OutputFileName");
  if (!fasciiFileName.empty()) {
    fasciiFile.open(fasciiFileName.c_str());
  } else {
    throw cms::Exception("Inconsistent Data") << " expected input file name is null\n";
  }
}

BeamSpotWrite2DB::~BeamSpotWrite2DB() = default;

void BeamSpotWrite2DB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

void BeamSpotWrite2DB::endJob() {
  edm::LogPrint("BeamSpotWrite2DB") << " Read beam spot data from text file: " << fasciiFileName;
  edm::LogPrint("BeamSpotWrite2DB") << " please see plugins/BeamSpotWrite2DB.cc for format of text file.";

  edm::LogInfo("BeamSpotWrite2DB")
      << " Content of the file is expected to have this format with the first column as a keyword:";
  edm::LogInfo("BeamSpotWrite2DB") << " x\n y\n z\n sigmaZ\n dxdz\n dydz\n beamWidthX\n beamWidthY";
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      edm::LogInfo("BeamSpotWrite2DB") << " cov[" << i << "][" << j << "] cov[" << i << "][" << j << "] cov[" << i
                                       << "][" << j << "] cov[" << i << "][" << j << "] cov[" << i << "][" << j
                                       << "] cov[" << j << "][" << j << "] cov[" << i << "][" << j << "]";
    }
  }

  // extract from file
  double x, y, z, sigmaZ, dxdz, dydz, beamWidthX, beamWidthY, emittanceX, emittanceY, betastar;
  std::string tag;
  double cov[7][7];
  int type;

  fasciiFile >> tag >> type;
  fasciiFile >> tag >> x;
  fasciiFile >> tag >> y;
  fasciiFile >> tag >> z;
  fasciiFile >> tag >> sigmaZ;
  fasciiFile >> tag >> dxdz;
  fasciiFile >> tag >> dydz;
  fasciiFile >> tag >> beamWidthX;
  fasciiFile >> tag >> beamWidthY;
  fasciiFile >> tag >> cov[0][0] >> cov[0][1] >> cov[0][2] >> cov[0][3] >> cov[0][4] >> cov[0][5] >> cov[0][6];
  fasciiFile >> tag >> cov[1][0] >> cov[1][1] >> cov[1][2] >> cov[1][3] >> cov[1][4] >> cov[1][5] >> cov[1][6];
  fasciiFile >> tag >> cov[2][0] >> cov[2][1] >> cov[2][2] >> cov[2][3] >> cov[2][4] >> cov[2][5] >> cov[2][6];
  fasciiFile >> tag >> cov[3][0] >> cov[3][1] >> cov[3][2] >> cov[3][3] >> cov[3][4] >> cov[3][5] >> cov[3][6];
  fasciiFile >> tag >> cov[4][0] >> cov[4][1] >> cov[4][2] >> cov[4][3] >> cov[4][4] >> cov[4][5] >> cov[4][6];
  fasciiFile >> tag >> cov[5][0] >> cov[5][1] >> cov[5][2] >> cov[5][3] >> cov[5][4] >> cov[5][5] >> cov[5][6];
  fasciiFile >> tag >> cov[6][0] >> cov[6][1] >> cov[6][2] >> cov[6][3] >> cov[6][4] >> cov[6][5] >> cov[6][6];
  fasciiFile >> tag >> emittanceX;
  fasciiFile >> tag >> emittanceY;
  fasciiFile >> tag >> betastar;

  BeamSpotObjects abeam;

  abeam.setType(type);
  abeam.setPosition(x, y, z);
  abeam.setSigmaZ(sigmaZ);
  abeam.setdxdz(dxdz);
  abeam.setdydz(dydz);
  abeam.setBeamWidthX(beamWidthX);
  abeam.setBeamWidthY(beamWidthY);
  abeam.setEmittanceX(emittanceX);
  abeam.setEmittanceY(emittanceY);
  abeam.setBetaStar(betastar);

  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      abeam.setCovariance(i, j, cov[i][j]);
    }
  }

  edm::LogPrint("BeamSpotWrite2DB") << " write results to DB...";

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    edm::LogPrint("BeamSpotWrite2DB") << "poolDBService available";
    if (poolDbService->isNewTagRequest("BeamSpotObjectsRcd")) {
      edm::LogPrint("BeamSpotWrite2DB") << "new tag requested";
      poolDbService->createOneIOV<BeamSpotObjects>(abeam, poolDbService->beginOfTime(), "BeamSpotObjectsRcd");
    } else {
      edm::LogPrint("BeamSpotWrite2DB") << "no new tag requested";
      poolDbService->appendOneIOV<BeamSpotObjects>(abeam, poolDbService->currentTime(), "BeamSpotObjectsRcd");
    }
  }
  edm::LogPrint("BeamSpotWrite2DB") << "[BeamSpotWrite2DB] endJob done \n";
}

void BeamSpotWrite2DB::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment(
      "Writes out a DB file containing a BeamSpotObjects payload, according to parameters defined in ASCII file");
  desc.addUntracked<std::string>("OutputFileName", {});
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotWrite2DB);
