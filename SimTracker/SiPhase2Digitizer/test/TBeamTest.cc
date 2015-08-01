// -*- C++ -*-
//
// Package:    TBeamTest
// Class:      TBeamTest
// 
/**\class TBeamTest TBeamTest.cc 

 Description: Access Digi collection and fill a few histograms to compare with TestBeam data

*/
//
// Author:  Suchandra Dutta, Suvankar RoyChoudhury
// Created:  July 2015
//
//
// system include files
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
class TBeamTest : public edm::EDAnalyzer {

public:

  explicit TBeamTest(const edm::ParameterSet&);
  ~TBeamTest();
  virtual void beginJob();
  virtual void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  virtual void endJob(); 

  struct DigiMEs{
    MonitorElement* NumberOfDigis;
    MonitorElement* PositionOfDigis;
    MonitorElement* NumberOfClusters;
    MonitorElement* ClusterWidth;
    MonitorElement* ClusterPosition;
  };

private:
  void bookHistos(unsigned int idet);
  uint32_t getTag(uint32_t idet);
  DQMStore* dqmStore_;
  edm::ParameterSet config_;
  std::map<uint32_t, DigiMEs> detMEs;
  edm::InputTag otDigiSrc_;
};
//
// constructors 
//
TBeamTest::TBeamTest(const edm::ParameterSet& iConfig) :
  dqmStore_(edm::Service<DQMStore>().operator->()),
  config_(iConfig)
{
  otDigiSrc_ = config_.getParameter<edm::InputTag>("OuterTrackerDigiSource");
  edm::LogInfo("TBeamTest") << ">>> Construct TBeamTest ";
}

//
// destructor
//
TBeamTest::~TBeamTest() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("TBeamTest")<< ">>> Destroy TBeamTest ";
}
//
// -- Begin Job
//
void TBeamTest::beginJob() {
   edm::LogInfo("TBeamTest")<< "Initialize TBeamTest ";
}
//
// -- Begin Run
//
void TBeamTest::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){
}
//
// -- Analyze
//
void TBeamTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;


  // Get digis

  edm::Handle< edm::DetSetVector<Phase2TrackerDigi> > otDigiHandle;
  iEvent.getByLabel(otDigiSrc_, otDigiHandle);

  const DetSetVector<Phase2TrackerDigi>* digis = otDigiHandle.product();

  edm::DetSetVector<Phase2TrackerDigi>::const_iterator DSViter;
  for(DSViter = digis->begin(); DSViter != digis->end(); DSViter++) {
    unsigned int rawid = DSViter->id; 
    DetId detId(rawid);
    edm::LogInfo("TBeamTest")<< " Det Id = " << rawid;    
    if (detId.subdetId() == PixelSubdetector::PixelBarrel || detId.subdetId() == PixelSubdetector::PixelEndcap) continue;    
    std::map<uint32_t, DigiMEs >::iterator pos = detMEs.find(getTag(rawid));
    if (pos == detMEs.end()) {
      bookHistos(rawid);
      pos = detMEs.find(getTag(rawid));
    } 
    DigiMEs local_mes = pos->second;
    int nDigi = 0; 
    int row_last = -1;
    int col_last = -1;
    int nclus = 0;
    int width = 0;
    int position = 0; 
    for (DetSet<Phase2TrackerDigi>::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
      int col = di->column(); // column
      int row = di->row();    // row
      nDigi++;
      std::cout <<  "  column " << col << " row " << row  << std::endl;
      edm::LogInfo("TBeamTest")<< "  column " << col << " row " << row  << std::endl;
      local_mes.PositionOfDigis->Fill(row+1);
      
      if (row_last == -1 ) {
        width  = 1;
        position = row+1;
        nclus++; 
      } else {
	if (abs(row - row_last) == 1 && col == col_last) {
	  position += row+1;
	  width++;
	} else {
          position /= width;  
          local_mes.ClusterWidth->Fill(width);
          local_mes.ClusterPosition->Fill(position);
	  width  = 1;
	  position = row+1;
          nclus++;
	}
      }
      edm::LogInfo("TBeamTest")<< " row " << row << " col " << col <<  " row_last " << row_last << " col_last " << col_last << " width " << width ;
      row_last = row;
      col_last = col;
    }
    local_mes.NumberOfClusters->Fill(nclus);  
    local_mes.NumberOfDigis->Fill(nDigi);
  }
}
//
// -- Book Histograms
//
void TBeamTest::bookHistos(unsigned int idet){ 
  std::map<uint32_t, DigiMEs >::iterator pos = detMEs.find(idet);
  if (pos == detMEs.end()) {

    std::stringstream folder_name;
   
    std::string top_folder = config_.getParameter<std::string>("TopFolderName");
    dqmStore_->cd();
    folder_name << top_folder <<"/det" << getTag(idet);

    edm::LogInfo("TBeamTest")<< " Booking Histograms in : " << folder_name.str();
    dqmStore_->setCurrentFolder(folder_name.str());

    std::ostringstream HistoName;

    DigiMEs local_mes;
    edm::ParameterSet Parameters =  config_.getParameter<edm::ParameterSet>("NumbeOfDigisH");
    HistoName.str("");
    HistoName << "NumberOfHits";
    local_mes.NumberOfDigis = dqmStore_->book1D(HistoName.str(), HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("PositionOfDigisH");
    HistoName.str("");
    HistoName << "HitPositions";
    local_mes.PositionOfDigis = dqmStore_->book1D(HistoName.str(), HistoName.str(),
						Parameters.getParameter<int32_t>("Nxbins"),
						Parameters.getParameter<double>("xmin"),
						  Parameters.getParameter<double>("xmax"));
    Parameters =  config_.getParameter<edm::ParameterSet>("NumberOfClustersH");
    HistoName.str("");
    HistoName << "NumberOfCluetsrs";
    local_mes.NumberOfClusters = dqmStore_->book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));
    Parameters =  config_.getParameter<edm::ParameterSet>("ClusterWidthH");
    HistoName.str("");
    HistoName << "ClusterWidth";
    local_mes.ClusterWidth = dqmStore_->book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));
    Parameters =  config_.getParameter<edm::ParameterSet>("ClusterPositionH");
    HistoName.str("");
    HistoName << "ClusterPositions";
    local_mes.ClusterPosition = dqmStore_->book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));
    detMEs.insert(std::make_pair(getTag(idet), local_mes)); 
  }  
}
uint32_t TBeamTest::getTag(unsigned int idet) {
  if (idet & (1 << 2)) return 0;
  else return 1;
}
//
// -- End Job
//
void TBeamTest::endJob(){
  dqmStore_->cd();
  dqmStore_->showDirStructure();  
}
//define this as a plug-in
DEFINE_FWK_MODULE(TBeamTest);
