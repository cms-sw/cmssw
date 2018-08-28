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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"


#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerFwd.h"

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"

//
// constructors 
//
Phase2TrackerMonitorDigi::Phase2TrackerMonitorDigi(const edm::ParameterSet& iConfig) :
  config_(iConfig),
  pixelFlag_(config_.getParameter<bool >("PixelPlotFillingFlag")),
  geomType_(config_.getParameter<std::string>("GeometryType")),
  otDigiSrc_(config_.getParameter<edm::InputTag>("OuterTrackerDigiSource")),
  itPixelDigiSrc_(config_.getParameter<edm::InputTag>("InnerPixelDigiSource")),
  otDigiToken_(consumes< edm::DetSetVector<Phase2TrackerDigi> >(otDigiSrc_)),
  itPixelDigiToken_(consumes< edm::DetSetVector<PixelDigi> >(itPixelDigiSrc_))
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
  iEvent.getByToken(itPixelDigiToken_, pixDigiHandle); 

  edm::Handle< edm::DetSetVector<Phase2TrackerDigi> > otDigiHandle;
  iEvent.getByToken(otDigiToken_, otDigiHandle); 

  // Tracker Topology 
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);

  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    edm::ESHandle<TrackerGeometry> geomHandle;
    iSetup.get<TrackerDigiGeometryRecord>().get(geomType_, geomHandle);

    if (pixelFlag_) fillITPixelDigiHistos(pixDigiHandle, geomHandle);
    else  fillOTDigiHistos(otDigiHandle, geomHandle);
  }
}
void Phase2TrackerMonitorDigi::fillITPixelDigiHistos(const edm::Handle<edm::DetSetVector<PixelDigi>>  handle, const edm::ESHandle<TrackerGeometry> gHandle) {
  const edm::DetSetVector<PixelDigi>* digis = handle.product();

  const TrackerTopology* tTopo = tTopoHandle_.product();
  const TrackerGeometry* tGeom = gHandle.product();  

  for (typename edm::DetSetVector<PixelDigi>::const_iterator DSViter = digis->begin(); DSViter != digis->end(); DSViter++) {
    unsigned int rawid = DSViter->id; 
    edm::LogInfo("Phase2TrackerMonitorDigi")<< " Det Id = " << rawid;    
    int layer = tTopo->getITPixelLayerNumber(rawid);
    if (layer < 0) continue;
    std::map<uint32_t, DigiMEs >::iterator pos = layerMEs.find(layer);
    if (pos == layerMEs.end()) continue;

    const DetId detId(rawid);

    if (DetId(detId).det() != DetId::Detector::Tracker) continue;
  
    const GeomDetUnit* gDetUnit = tGeom->idToDetUnit(detId);
    const GeomDet *geomDet = tGeom->idToDet(detId);

    const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(gDetUnit);
    int nRows     = tkDetUnit->specificTopology().nrows();
    int nColumns  = tkDetUnit->specificTopology().ncolumns();
    if (nRows*nColumns == 0) continue;  

    DigiMEs& local_mes = pos->second;

    local_mes.nHitDetsPerLayer++;

    int nDigi = 0; 
    int row_last = -1;
    int col_last = -1;
    int nclus = 0;
    int width = 1;
    int position = 0; 
    std::vector<int> charges;
    for (typename edm::DetSet< PixelDigi >::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
      int col = di->column(); // column
      int row = di->row();    // row
      int adc = di->adc();    // digi charge 
      if (geomDet) {
        MeasurementPoint mp( row + 0.5, col + 0.5 );
        GlobalPoint pdPos = geomDet->surface().toGlobal( gDetUnit->topology().localPosition( mp ) ) ;
	XYPositionMap->Fill(pdPos.x()*10., pdPos.y()*10.);
	RZPositionMap->Fill(pdPos.z()*10., std::hypot(pdPos.x(),pdPos.y())*10.);  
      }
      nDigi++;
      edm::LogInfo("Phase2TrackerMonitorDigi")<< "  column " << col << " row " << row  <<
        std::dec  << std::endl;
      local_mes.PositionOfDigis->Fill(row+1, col+1);
      if (local_mes.ChargeOfDigis) local_mes.ChargeOfDigis->Fill(adc);
      if (row_last == -1 ) {
        position = row+1;
        nclus++; 
        charges.push_back(adc);
      } else {
	if (abs(row - row_last) == 1 && col == col_last) {
	  position += row+1;
	  width++;
	  charges.push_back(adc);
	} else {
          position /= width;  
          local_mes.ClusterWidth->Fill(width);
          local_mes.ClusterPosition->Fill(position);
	  for (auto v: charges) local_mes.ChargeOfDigisVsWidth->Fill(v, width);
	  charges.clear();
          charges.push_back(adc);
	  width  = 1;
	  position = row+1;
          nclus++;
	}
      }
      edm::LogInfo("Phase2TrackerMonitorDigi")<< " row " << row << " col " << col <<  " row_last " << row_last << " col_last " << col_last << " width " << width;
      row_last = row;
      col_last = col;
    }
    local_mes.NumberOfClustersPerDet->Fill(nclus);  
    local_mes.NumberOfDigisPerDet->Fill(nDigi);
    local_mes.nDigiPerLayer += nDigi;
    float occupancy = 1.0;
    if (nRows*nColumns > 0) occupancy = nDigi*1.0/(nRows*nColumns);
    if (geomDet) {
      GlobalPoint gp = geomDet->surface().toGlobal( gDetUnit->topology().localPosition( MeasurementPoint(nRows/2.0,nColumns/2.0))) ;
      XYOccupancyMap->Fill(gp.x()*10., gp.y()*10, occupancy);
      RZOccupancyMap->Fill(gp.z()*10., std::hypot(gp.x(),gp.y())*10., occupancy);  
      local_mes.EtaOccupancyProfP->Fill(gp.eta(), occupancy);
    }
    local_mes.DigiOccupancyP->Fill(occupancy);
  }
  // Fill histograms after loop over digis are complete
  for ( auto& ilayer : layerMEs) {
    DigiMEs& local_mes = ilayer.second;
    local_mes.TotalNumberOfDigisPerLayer->Fill(local_mes.nDigiPerLayer);
    local_mes.NumberOfHitDetectorsPerLayer->Fill(local_mes.nHitDetsPerLayer);
    local_mes.nDigiPerLayer = 0;
    local_mes.nHitDetsPerLayer = 0;
  }
}
void Phase2TrackerMonitorDigi::fillOTDigiHistos(const edm::Handle<edm::DetSetVector<Phase2TrackerDigi>>  handle, const edm::ESHandle<TrackerGeometry> gHandle) {
  const edm::DetSetVector<Phase2TrackerDigi>* digis = handle.product();

  const TrackerTopology* tTopo = tTopoHandle_.product();
  const TrackerGeometry* tGeom = gHandle.product();  

  for (typename edm::DetSetVector<Phase2TrackerDigi>::const_iterator DSViter = digis->begin(); DSViter != digis->end(); DSViter++) {
    unsigned int rawid = DSViter->id; 
    DetId detId(rawid);
    edm::LogInfo("Phase2TrackerMonitorDigi")<< " Det Id = " << rawid;    
    int layer = tTopo->getOTLayerNumber(rawid);
    if (layer < 0) continue;
    std::map<uint32_t, DigiMEs >::iterator pos = layerMEs.find(layer);
    if (pos == layerMEs.end()) continue;
    DigiMEs& local_mes = pos->second;

    local_mes.nHitDetsPerLayer++;
    if (DetId(detId).det() != DetId::Detector::Tracker) continue;
  
    const GeomDetUnit* gDetUnit = tGeom->idToDetUnit(detId);
    const GeomDet *geomDet = tGeom->idToDet(detId);

    const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(gDetUnit);
    int nRows     = tkDetUnit->specificTopology().nrows();
    int nColumns  = tkDetUnit->specificTopology().ncolumns();
    if (nRows*nColumns == 0) continue;  

    int nDigi = 0; 
    int row_last = -1;
    int col_last = -1;
    int nclus = 0;
    int width = 1;
    int position = 0; 
    float frac_ot = 0.;
    for (typename edm::DetSet< Phase2TrackerDigi >::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
      int col = di->column(); // column
      int row = di->row();    // row
      const DetId detId(rawid);

      if (geomDet) {
        MeasurementPoint mp( row + 0.5, col + 0.5 );
        GlobalPoint pdPos = geomDet->surface().toGlobal( gDetUnit->topology().localPosition( mp ) ) ;
	XYPositionMap->Fill(pdPos.x()*10., pdPos.y()*10.);
	RZPositionMap->Fill(pdPos.z()*10., std::hypot(pdPos.x(),pdPos.y())*10.);  
      }
      nDigi++;
      if (di->overThreshold()) frac_ot++;
      edm::LogInfo("Phase2TrackerMonitorDigi")<< "  column " << col << " row " << row  <<
        std::dec  << std::endl;
      local_mes.PositionOfDigis->Fill(row+1, col+1);


      if (row_last == -1 ) {
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
      edm::LogInfo("Phase2TrackerMonitorDigi")<< " row " << row << " col " << col <<  " row_last " << row_last << " col_last " << col_last << " width " << width;
      row_last = row;
      col_last = col;
    }
    local_mes.NumberOfClustersPerDet->Fill(nclus);  
    local_mes.NumberOfDigisPerDet->Fill(nDigi);
    local_mes.nDigiPerLayer += nDigi;
    if (nDigi) frac_ot /= nDigi;
    if (local_mes.FractionOfOvTBits && nColumns <= 2) local_mes.FractionOfOvTBits->Fill(frac_ot);

    float occupancy = 1.0;
    if (nRows*nColumns > 0) occupancy = nDigi*1.0/(nRows*nColumns);
    if (geomDet) {
      GlobalPoint gp = geomDet->surface().toGlobal( gDetUnit->topology().localPosition( MeasurementPoint(0.0,0.0))) ;
      XYOccupancyMap->Fill(gp.x()*10., gp.y()*10., occupancy);
      RZOccupancyMap->Fill(gp.z()*10., std::hypot(gp.x(),gp.y())*10., occupancy);  
      if (nColumns > 2) {
	if (local_mes.DigiOccupancyP) local_mes.DigiOccupancyP->Fill(occupancy);
	if (local_mes.EtaOccupancyProfP) local_mes.EtaOccupancyProfP->Fill(gp.eta(), occupancy);
      } else {
	local_mes.DigiOccupancyS->Fill(occupancy);
	local_mes.EtaOccupancyProfS->Fill(gp.eta(), occupancy);
	if (local_mes.FractionOfOvTBitsVsEta) local_mes.FractionOfOvTBitsVsEta->Fill(gp.eta(), frac_ot);

      }
    }
  }
  // Fill histograms after loop over digis are complete
  for (auto& ilayer : layerMEs) {
    DigiMEs& local_mes = ilayer.second;
    local_mes.TotalNumberOfDigisPerLayer->Fill(local_mes.nDigiPerLayer);
    local_mes.NumberOfHitDetectorsPerLayer->Fill(local_mes.nHitDetsPerLayer);
    local_mes.nDigiPerLayer = 0;
    local_mes.nHitDetsPerLayer = 0;
  }
}
//
// -- Book Histograms
//
void Phase2TrackerMonitorDigi::bookHistograms(DQMStore::IBooker & ibooker,
		 edm::Run const &  iRun ,
		 edm::EventSetup const &  iSetup ) {

  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;

  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);
  const TrackerTopology* const tTopo = tTopoHandle_.product();

  if (theTkDigiGeomWatcher.check(iSetup)) {
    edm::ESHandle<TrackerGeometry> geom_handle;
    iSetup.get<TrackerDigiGeometryRecord>().get(geomType_, geom_handle);
    const TrackerGeometry* tGeom = geom_handle.product();  
    for (auto const & det_u : tGeom->detUnits()) {
      unsigned int detId_raw = det_u->geographicalId().rawId();
      bookLayerHistos(ibooker,detId_raw, tTopo); 
    }
  }
  ibooker.cd();
  std::stringstream folder_name;
  folder_name << top_folder << "/" << "DigiMonitor";
  ibooker.setCurrentFolder(folder_name.str());

  edm::ParameterSet Parameters =  config_.getParameter<edm::ParameterSet>("XYPositionMapH");  
  edm::ParameterSet ParametersOcc =  config_.getParameter<edm::ParameterSet>("DigiOccupancyPH");
  XYPositionMap = ibooker.book2D("DigiXPosVsYPos","DigiXPosVsYPos",
				 Parameters.getParameter<int32_t>("Nxbins"),
				 Parameters.getParameter<double>("xmin"),
				 Parameters.getParameter<double>("xmax"),
				 Parameters.getParameter<int32_t>("Nybins"),
				 Parameters.getParameter<double>("ymin"),
				 Parameters.getParameter<double>("ymax"));
  XYOccupancyMap = ibooker.bookProfile2D("OccupancyInXY","OccupancyInXY",
					 Parameters.getParameter<int32_t>("Nxbins"),
					 Parameters.getParameter<double>("xmin"),
					 Parameters.getParameter<double>("xmax"),
					 Parameters.getParameter<int32_t>("Nybins"),
					 Parameters.getParameter<double>("ymin"),
					 Parameters.getParameter<double>("ymax"),
					 ParametersOcc.getParameter<double>("xmin"),
					 ParametersOcc.getParameter<double>("xmax"));

  Parameters =  config_.getParameter<edm::ParameterSet>("RZPositionMapH");  
  RZPositionMap = ibooker.book2D("DigiRPosVsZPos","DigiRPosVsZPos",
				 Parameters.getParameter<int32_t>("Nxbins"),
				 Parameters.getParameter<double>("xmin"),
				 Parameters.getParameter<double>("xmax"),
				 Parameters.getParameter<int32_t>("Nybins"),
				 Parameters.getParameter<double>("ymin"),
				 Parameters.getParameter<double>("ymax"));
  RZOccupancyMap = ibooker.bookProfile2D("OccupancyInRZ","OccupancyInRZ",
					 Parameters.getParameter<int32_t>("Nxbins"),
					 Parameters.getParameter<double>("xmin"),
					 Parameters.getParameter<double>("xmax"),
					 Parameters.getParameter<int32_t>("Nybins"),
					 Parameters.getParameter<double>("ymin"),
					 Parameters.getParameter<double>("ymax"),
					 ParametersOcc.getParameter<double>("xmin"),
					 ParametersOcc.getParameter<double>("xmax"));
}
//
// -- Book Layer Histograms
//
void Phase2TrackerMonitorDigi::bookLayerHistos(DQMStore::IBooker & ibooker, unsigned int det_id, const TrackerTopology* tTopo){ 

  
  int layer;
  if (pixelFlag_) layer = tTopo->getITPixelLayerNumber(det_id);
  else layer = tTopo->getOTLayerNumber(det_id);

  if (layer < 0) return;
  std::map<uint32_t, DigiMEs >::iterator pos = layerMEs.find(layer);
  if (pos == layerMEs.end()) {

    std::string top_folder = config_.getParameter<std::string>("TopFolderName");
    std::stringstream folder_name;

    // initialise Histograms

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

    local_mes.nDigiPerLayer = 0;
    local_mes.nHitDetsPerLayer = 0;

    edm::ParameterSet Parameters =  config_.getParameter<edm::ParameterSet>("NumberOfDigisPerDetH");
    edm::ParameterSet EtaParameters =  config_.getParameter<edm::ParameterSet>("EtaH");
    HistoName.str("");
    HistoName << "NumberOfDigisPerDet_" << fname2.str();
    local_mes.NumberOfDigisPerDet = ibooker.book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));
    Parameters =  config_.getParameter<edm::ParameterSet>("DigiOccupancyPH");
    HistoName.str("");
    HistoName << "DigiOccupancyP_" << fname2.str();
    local_mes.DigiOccupancyP = ibooker.book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "DigiOccupancyVsEtaP_" << fname2.str();
    local_mes.EtaOccupancyProfP = ibooker.bookProfile(HistoName.str(), HistoName.str(), 
	  EtaParameters.getParameter<int32_t>("Nbins"),EtaParameters.getParameter<double>("xmin"),EtaParameters.getParameter<double>("xmax"),
          Parameters.getParameter<double>("xmin"),Parameters.getParameter<double>("xmax"),"");

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

    Parameters =  config_.getParameter<edm::ParameterSet>("TotalNumberOfDigisPerLayerH");
    HistoName.str("");
    HistoName << "TotalNumberOfDigisPerLayer_" << fname2.str();
    local_mes.TotalNumberOfDigisPerLayer = ibooker.book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("NumberOfHitDetsPerLayerH");
    HistoName.str("");
    HistoName << "NumberOfHitDetectorsPerLayer_" << fname2.str();
    local_mes.NumberOfHitDetectorsPerLayer = ibooker.book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("NumberOfClustersPerDetH");
    HistoName.str("");
    HistoName << "NumberOfClustersPerDet_" << fname2.str();
    local_mes.NumberOfClustersPerDet = ibooker.book1D(HistoName.str(), HistoName.str(),
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
    Parameters =  config_.getParameter<edm::ParameterSet>("ClusterPositionH");
    HistoName.str("");
    HistoName << "ClusterPosition_" << fname2.str();
    local_mes.ClusterPosition = ibooker.book1D(HistoName.str(), HistoName.str(),
					     Parameters.getParameter<int32_t>("Nbins"),
					     Parameters.getParameter<double>("xmin"),
					     Parameters.getParameter<double>("xmax"));
   
    if (!pixelFlag_) {

      Parameters =  config_.getParameter<edm::ParameterSet>("DigiOccupancySH");
      HistoName.str("");
      HistoName << "DigiOccupancyS_" << fname2.str();
      local_mes.DigiOccupancyS = ibooker.book1D(HistoName.str(), HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
      
      HistoName.str("");
      HistoName << "DigiOccupancyVsEtaS_" << fname2.str();
      local_mes.EtaOccupancyProfS = ibooker.bookProfile(HistoName.str(), HistoName.str(),
        EtaParameters.getParameter<int32_t>("Nbins"),EtaParameters.getParameter<double>("xmin"),EtaParameters.getParameter<double>("xmax"),
           Parameters.getParameter<double>("xmin"),Parameters.getParameter<double>("xmax"),"");

      HistoName.str("");
      HistoName << "FractionOfOverThresholdDigis_" << fname2.str();
      local_mes.FractionOfOvTBits= ibooker.book1D(HistoName.str(), HistoName.str(),11, -0.05, 1.05);

      edm::ParameterSet Parameters =  config_.getParameter<edm::ParameterSet>("NumberOfDigisPerDetH");
      HistoName.str("");
      HistoName << "FractionOfOverThresholdDigisVaEta_" << fname2.str();
      local_mes.FractionOfOvTBitsVsEta= ibooker.bookProfile(HistoName.str(), HistoName.str(), 	
							    EtaParameters.getParameter<int32_t>("Nbins"),EtaParameters.getParameter<double>("xmin"),EtaParameters.getParameter<double>("xmax"),
							    Parameters.getParameter<double>("xmin"),Parameters.getParameter<double>("xmax"),"");
    } else {

      Parameters =  config_.getParameter<edm::ParameterSet>("DigiChargeH");
      HistoName.str("");
      HistoName << "ChargeOfDigis_" << fname2.str();   
      local_mes.ChargeOfDigis = ibooker.book1D(HistoName.str(),HistoName.str(),
					       Parameters.getParameter<int32_t>("Nbins"),
					       Parameters.getParameter<double>("xmin"),
					       Parameters.getParameter<double>("xmax"));

      edm::ParameterSet WidthParameters =   config_.getParameter<edm::ParameterSet>("ClusterWidthH");
      HistoName.str("");
      HistoName << "ChargeOfDigisVsWidth_" << fname2.str();   
      local_mes.ChargeOfDigisVsWidth = ibooker.book2D(HistoName.str(),HistoName.str(),
					       Parameters.getParameter<int32_t>("Nbins"),
					       Parameters.getParameter<double>("xmin"),
					       Parameters.getParameter<double>("xmax"),
					       WidthParameters.getParameter<int32_t>("Nbins"),
					       WidthParameters.getParameter<double>("xmin"),
					       WidthParameters.getParameter<double>("xmax"));


    }

    layerMEs.insert(std::make_pair(layer, local_mes)); 
  }  
}
void Phase2TrackerMonitorDigi::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup){
}
//define this as a plug-in
DEFINE_FWK_MODULE(Phase2TrackerMonitorDigi);
