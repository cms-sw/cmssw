// -*- C++ -*-
//
// Package:    Phase2TrackerMonitorDigi
// Class:      Phase2TrackerMonitorDigi
// 
/**\class Phase2TrackerMonitorDigi Phase2TrackerMonitorDigi.cc 

 Description: Test pixel digis. 

*/
//
// Author: Suchandra Dutta, Suvankar Roy Chowdhury, Subir Sarkar
// Date: January 29, 2016
//
//
// system include files
#include <memory>
#include "SimTracker/SiPhase2Digitizer/test/Phase2TrackerMonitorDigi.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"


#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"

//
// constructors 
//
Phase2TrackerMonitorDigi::Phase2TrackerMonitorDigi(const edm::ParameterSet& iConfig) :
  config_(iConfig),
  pixDigiSrc_(config_.getParameter<edm::InputTag>("PixelDigiSource")),
  otDigiSrc_(config_.getParameter<edm::InputTag>("OuterTrackerDigiSource")),
  pixDigiToken_(consumes< edm::DetSetVector<PixelDigi> >(pixDigiSrc_)),
  otDigiToken_(consumes< edm::DetSetVector<Phase2TrackerDigi> >(otDigiSrc_))
{
  edm::LogInfo("Phase2TrackerMonitorDigi") << ">>> Construct Phase2TrackerMonitorDigi ";
}

//
// destructor
//
Phase2TrackerMonitorDigi::~Phase2TrackerMonitorDigi() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("Phase2TrackerMonitorDigi")<< ">>> Destroy Phase2TrackerMonitorDigi ";
}
//
// -- DQM Begin Run 
//
void Phase2TrackerMonitorDigi::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
   edm::LogInfo("Phase2TrackerMonitorDigi")<< "Initialize Phase2TrackerMonitorDigi ";
}
// -- Analyze
//
void Phase2TrackerMonitorDigi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;


  // Get digis
  edm::Handle< edm::DetSetVector<PixelDigi> > pixDigiHandle;
  iEvent.getByToken(pixDigiToken_ , pixDigiHandle); 
  //iEvent.getByLabel(pixDigiSrc_, pixDigiHandle);

  edm::Handle< edm::DetSetVector<Phase2TrackerDigi> > otDigiHandle;
  iEvent.getByToken(otDigiToken_ , otDigiHandle); 
  //iEvent.getByLabel(otDigiSrc_, otDigiHandle);

  const DetSetVector<Phase2TrackerDigi>* digis = otDigiHandle.product();

  // Tracker Topology 
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* tTopo = tTopoHandle.product();

  edm::DetSetVector<Phase2TrackerDigi>::const_iterator DSViter;
  for(DSViter = digis->begin(); DSViter != digis->end(); DSViter++) {
    unsigned int rawid = DSViter->id; 
    DetId detId(rawid);
    edm::LogInfo("Phase2TrackerMonitorDigi")<< " Det Id = " << rawid;    
    int layer = tTopo->getOTLayerNumber(rawid);
    if (layer < 0) continue;
    std::map<uint32_t, DigiMEs >::iterator pos = layerMEs.find(layer);
    if (pos == layerMEs.end()) continue;
    DigiMEs local_mes = pos->second;
    int nDigi = 0; 
    int row_last = -1;
    int col_last = -1;
    int nclus = 0;
    int width = 0;
    int position = 0; 
    unsigned short charge = 0; 
    for (DetSet<Phase2TrackerDigi>::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
      //      unsigned short adc = di->adc();    // charge, modified to unsiged short
      unsigned short adc = 255;    // charge, modified to unsiged short
      int col = di->column(); // column
      int row = di->row();    // row
      nDigi++;
      edm::LogInfo("Phase2TrackerMonitorDigi")<< "  column " << col << " row " << row << " ADC " << adc <<
        std::dec  << std::endl;
      local_mes.PositionOfDigis->Fill(row+1, col+1);
      local_mes.DigiCharge->Fill(adc);
      
      if (row_last == -1 ) {
        charge = adc;
        width  = 1;
        position = row+1;
        nclus++; 
      } else {
	if (abs(row - row_last) == 1 && col == col_last) {
	  charge += adc;
	  position += row+1;
	  width++;
	} else {
          position /= width;  
          local_mes.ClusterCharge->Fill(charge);
          local_mes.ClusterWidth->Fill(width);
          local_mes.ClusterPosition->Fill(position);
	  charge = adc;
	  width  = 1;
	  position = row+1;
          nclus++;
	}
      }
      edm::LogInfo("Phase2TrackerMonitorDigi")<< " row " << row << " col " << col <<  " row_last " << row_last << " col_last " << col_last << " width " << width << " charge " << charge;
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
void Phase2TrackerMonitorDigi::bookHistograms(DQMStore::IBooker & ibooker,
		 edm::Run const &  iRun ,
		 edm::EventSetup const &  iSetup ) {

  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  std::string geometry_type = config_.getParameter<std::string>("GeometryType");
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;

  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  if (theTkDigiGeomWatcher.check(iSetup)) {
    edm::ESHandle<TrackerGeometry> geom_handle;
    iSetup.get<TrackerDigiGeometryRecord>().get(geometry_type, geom_handle);
    for (auto const & det_u : geom_handle->detUnits()) {
      unsigned int detId_raw = det_u->geographicalId().rawId();
      bookLayerHistos(ibooker,detId_raw, tTopo); 
    }
  }
  ibooker.cd();
  ibooker.setCurrentFolder(top_folder);
}
//
// -- Book Layer Histograms
//
void Phase2TrackerMonitorDigi::bookLayerHistos(DQMStore::IBooker & ibooker, unsigned int det_id, const TrackerTopology* tTopo){ 

  int layer = tTopo->getOTLayerNumber(det_id);
  if (layer < 0) return;
  std::map<uint32_t, DigiMEs >::iterator pos = layerMEs.find(layer);
  if (pos == layerMEs.end()) {

    std::string top_folder = config_.getParameter<std::string>("TopFolderName");
    std::stringstream folder_name;

    std::ostringstream fname1, fname2, tag;
    if (layer < 100) { 
      fname1 << "Barrel";
      fname2 << "Layer_" << layer;    
    } else {
      int side = layer/100;
      int idisc = layer - side*100; 
      fname1 << "EndCap_Side_" << side; 
      fname2 << "Disc_" << idisc;       
    }
    
    ibooker.cd();
    folder_name << top_folder << "/" << "DigiMonitor" << "/"<< fname1.str() << "/" << fname2.str() ;
    ibooker.setCurrentFolder(folder_name.str());    

    edm::LogInfo("Phase2TrackerMonitorDigi")<< " Booking Histograms in : " << folder_name.str();

    std::ostringstream HistoName;

    DigiMEs local_mes;
    edm::ParameterSet Parameters =  config_.getParameter<edm::ParameterSet>("NumbeOfDigisH");
    HistoName.str("");
    HistoName << "NumberOfDigis_" << fname2.str();
    local_mes.NumberOfDigis = ibooker.book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("PositionOfDigisH");
    HistoName.str("");
    HistoName << "PositionOfDigis_" << fname2.str().c_str();
    local_mes.PositionOfDigis = ibooker.book2D(HistoName.str(), HistoName.str(),
					       Parameters.getParameter<int32_t>("Nxbins"),
					       Parameters.getParameter<double>("xmin"),
					       Parameters.getParameter<double>("xmax"),
					       Parameters.getParameter<int32_t>("Nybins"),
					       Parameters.getParameter<double>("ymin"),
					       Parameters.getParameter<double>("ymax"));
    Parameters =  config_.getParameter<edm::ParameterSet>("DigiChargeH");
    HistoName.str("");
    HistoName << "DigiCharge_" << fname2.str();
    local_mes.DigiCharge = ibooker.book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("NumberOfClustersH");
    HistoName.str("");
    HistoName << "NumberOfClusters_" << fname2.str();
    local_mes.NumberOfClusters = ibooker.book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));
    Parameters =  config_.getParameter<edm::ParameterSet>("ClusterWidthH");
    HistoName.str("");
    HistoName << "ClusterWidth_" << fname2.str();
    local_mes.ClusterWidth = ibooker.book1D(HistoName.str(), HistoName.str(),
					    Parameters.getParameter<int32_t>("Nbins"),
					    Parameters.getParameter<double>("xmin"),
					    Parameters.getParameter<double>("xmax"));
    Parameters =  config_.getParameter<edm::ParameterSet>("ClusterChargeH");
    HistoName.str("");
    HistoName << "ClusterCharge_" << fname2.str();
    local_mes.ClusterCharge = ibooker.book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));
    Parameters =  config_.getParameter<edm::ParameterSet>("ClusterPositionH");
    HistoName.str("");
    HistoName << "ClusterPosition_" << fname2.str();
    local_mes.ClusterPosition = ibooker.book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));
    layerMEs.insert(std::make_pair(layer, local_mes)); 
  }  
}
void Phase2TrackerMonitorDigi::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup){
}
//define this as a plug-in
DEFINE_FWK_MODULE(Phase2TrackerMonitorDigi);
