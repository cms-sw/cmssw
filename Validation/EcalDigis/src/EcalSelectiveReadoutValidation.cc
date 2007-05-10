/*
 * \file EcalSelectiveReadoutValidation.cc
 *
 * $Date: 2007/03/20 13:06:45 $
 * $Revision: 1.20 $
 *
*/

#include "Validation/EcalDigis/interface/EcalSelectiveReadoutValidation.h"

#include "Validation/EcalDigis/src/ecalDccMap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
//#include "DataFormats/EcalDetId/interface/EcalScDetId.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include <string.h>

using namespace cms;
using namespace edm;
using namespace std;

EcalSelectiveReadoutValidation::EcalSelectiveReadoutValidation(const ParameterSet& ps):
  ebDigiCollection_(ps.getParameter<edm::InputTag>("EbDigiCollection")),
  eeDigiCollection_(ps.getParameter<edm::InputTag>("EeDigiCollection")),
  ebSrFlagCollection_(ps.getParameter<edm::InputTag>("EbSrFlagCollection")),
  eeSrFlagCollection_(ps.getParameter<edm::InputTag>("EeSrFlagCollection")),
  triggerTowerMap_(0){
  
  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  
  if(outputFile_.size() != 0){
    LogInfo("OutputInfo") << " Ecal Digi Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else{
    LogInfo("OutputInfo") << " Ecal Digi Task histograms will NOT be saved";
  }
 
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
  
  if(verbose_){
    cout << " verbose switch is ON" << endl;
  } else{
    cout << " verbose switch is OFF" << endl;
  }
  
  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  if(verbose_){
    dbe_->setVerbose(1);
  } else{
    dbe_->setVerbose(0);
  }

  if(verbose_) dbe_->showDirStructure();
  
  dbe_->setCurrentFolder("EcalDigiTask");
  
  meDccVol_ = dbe_->bookProfile("dccVol",
				"DCC event fragment size;Dcc id; "
				"<Event size> (kB)", 54, .5, 54.5,
				0, 0, 0);
  
  meVolBLI_ = dbe_->book1D("volBLI",
			   "Barrel low interest data volume;"
			   "Event size (kB);Nevts",
			   100, 0., 200.);
  meVolELI_ = dbe_->book1D("volELI",
			   "Endcap low interest data volume;"
			   "Event size (kB);Nevts",
			   100, 0., 200.);
  meVolLI_ = dbe_->book1D("volLI",
			  "ECAL low interest data volume;"
			  "Event size (kB);Nevts",
			  100, 0., 200.);
  meVolBHI_ = dbe_->book1D("volBHI",
			   "Barrel high interest data volume;"
			   "Event size (kB);Nevts",
			   100, 0., 200.);
  meVolEHI_ = dbe_->book1D("volEHI",
		 "Endcap high interest data volume;"
			   "Event size (kB);Nevts",
			   100, 0., 200.);
  
  meVolHI_ = dbe_->book1D("volHI",
			  "ECAL high interest data volume;"
			  "Event size (kB);Nevts",
			  100, 0., 200.);
  
  meVolB_ = dbe_->book1D("volB",
			 "Barrel data volume;Event size (kB);Nevts",
			 100, 0., 200.);
  
  meVolE_ = dbe_->book1D("volE",
			 "Endcap data volume;Event size (kB);Nevts",
			 100, 0., 200.);
  
  meVol_ = dbe_->book1D("vol",
			"ECAL data volume;Event size (kB);Nevts",
			100, 0., 200.);
}

EcalSelectiveReadoutValidation::~EcalSelectiveReadoutValidation(){
 
  if(outputFile_.size()!=0) dbe_->save(outputFile_);

}

void EcalSelectiveReadoutValidation::beginJob(const EventSetup& eventSetup){
  // endcap mapping
  edm::ESHandle<EcalTrigTowerConstituentsMap> hTriggerTowerMap;
  eventSetup.get<IdealGeometryRecord>().get(hTriggerTowerMap);
  triggerTowerMap_ = hTriggerTowerMap.product();
}

void EcalSelectiveReadoutValidation::endJob(){
}

void EcalSelectiveReadoutValidation::analyze(const Event& e,
					     const EventSetup& c){
  
  LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = "
		       << e.id().event();

  Handle<EBDigiCollection> hEbDigis;
  Handle<EEDigiCollection> hEeDigis;
  Handle<EBSrFlagCollection> hEbSrFlags;
  Handle<EESrFlagCollection> hEeSrFlags;
  
  const EBDigiCollection* ebDigis = 0;
  const EEDigiCollection* eeDigis = 0;
  const EBSrFlagCollection* ebSrFlags = 0;
  const EESrFlagCollection* eeSrFlags = 0;

  try{
    e.getByLabel(ebDigiCollection_, hEbDigis);
    ebDigis = hEbDigis.product();
    LogDebug("DigiInfo") << "total # EBdigis: " << ebDigis->size() ;
  } catch(cms::Exception &e){
    LogWarning("DigiInfo") << "EB digis no fount";
  }
  
  try{
    e.getByLabel(eeDigiCollection_, hEeDigis);
    eeDigis = hEeDigis.product();
    LogDebug("DigiInfo") << "total # EEdigis: " << eeDigis->size() ;
  } catch(cms::Exception &e){
    LogWarning("DigiInfo") << "EE digis no fount";
  }

  try{
    e.getByLabel(ebSrFlagCollection_, hEbSrFlags);
    ebSrFlags = hEbSrFlags.product();
    LogDebug("DigiInfo") << "total # EB SR Flags: " << ebSrFlags->size() ;
  } catch(cms::Exception &e){
    LogWarning("DigiInfo") << "EB SR Flags no fount";
  }

  try{
    e.getByLabel(eeSrFlagCollection_, hEeSrFlags);
    eeSrFlags = hEeSrFlags.product();
    LogDebug("DigiInfo") << "total # EE SR Flags: " << eeSrFlags->size() ;
  } catch(cms::Exception &e){
    LogWarning("DigiInfo") << "EE SR Flags no fount";
  }

  
  anaDigiInit();

  //Barrel
  if(ebDigis && ebSrFlags){
    for(std::vector<EBDataFrame>::const_iterator it = ebDigis->begin() ;
	it != ebDigis->end() ;
	++it){
      anaDigi(*it, *ebSrFlags);
    }
  }
  
  // Endcap
  if(eeDigis && eeSrFlags){
    for(std::vector<EEDataFrame>::const_iterator it = eeDigis->begin() ;
	it != eeDigis->end() ;
	++it){
      anaDigi(*it, *eeSrFlags);
    }
  }

  //histos
  for(unsigned iDcc = 0; iDcc <  nDccs; ++iDcc){ 
    meDccVol_->Fill(iDcc, getDccEventSize(iDcc, nPerDcc_[iDcc]));	
  }
  
  double a = getEbEventSize(nEbLI_)/kByte_;
  meVolBLI_->Fill(a);
  double b = getEeEventSize(nEeLI_)/kByte_;
  meVolELI_->Fill(b);	
  meVolLI_->Fill(a+b);	
  
  a = getEbEventSize(nEbLI_)/kByte_;
  meVolBHI_->Fill(a);
  b = getEeEventSize(nEeLI_)/kByte_;
  meVolEHI_->Fill(b);	
  meVolHI_->Fill(a+b);

  a = getEbEventSize(nEb_)/kByte_;
  meVolB_->Fill(a);
  b = getEbEventSize(nEe_)/kByte_;
  meVolE_->Fill(b);
  meVol_->Fill(a+b);
}


template<class T, class U>
void EcalSelectiveReadoutValidation::anaDigi(const T& frame,
					     const U& srFlagColl){
  const DetId& xtalId = frame.id();
  typename U::const_iterator srf = srFlagColl.find(readOutUnitOf(frame.id()));
  
  if(srf == srFlagColl.end()){
    throw cms::Exception("");
  }
  
  bool highInterest = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK)
		       == EcalSrFlag::SRF_FULL);
  bool barrel = (xtalId.subdetId()==EcalBarrel);

  if(barrel){
    ++nEb_;
    if(highInterest){
      ++nEbLI_;
    } else{//low interest
      ++nEbHI_;
    }
  } else{//endcap
    ++nEe_;
    if(highInterest){
      ++nEeLI_;
    } else{//low interest
      ++nEeHI_;
    }
  }

  ++nPerDcc_[dccNum(xtalId)-1];
}

void EcalSelectiveReadoutValidation::anaDigiInit(){
  nEb_ = 0;
  nEe_ = 0;
  nEeLI_ = 0;
  nEeHI_ = 0;
  nEbLI_ = 0;
  nEbHI_ = 0;
  bzero(nPerDcc_, sizeof(nPerDcc_));
}

int EcalSelectiveReadoutValidation::getRuCount(int iDcc0) const{
  static int nEemRu[] = {34, 32, 33, 33, 32, 34, 33, 34, 33};
  static int nEepRu[] = {32, 33, 33, 32, 34, 33, 34, 33, 34};
  if(iDcc0<9){//EE-
    return nEemRu[iDcc0];
  } else if(iDcc0>=45){//EE+
    return nEepRu[iDcc0-45];
  } else{//EB
    return 68;
  }
}

unsigned EcalSelectiveReadoutValidation::dccNum(const DetId& xtalId) const{
  int i;
  int j;
  int k;
  
  assert(xtalId.det()==DetId::Ecal);
  assert(!xtalId.null());
  
  if(xtalId.subdetId()==EcalBarrel){
    EBDetId ebDetId(xtalId);
    i = 1; //barrel
    j = iEta2cIndex(ebDetId.ieta());
    k = iPhi2cIndex(ebDetId.iphi());
  } else if(xtalId.subdetId()==EcalEndcap){
    EEDetId eeDetId(xtalId);
    i = eeDetId.zside()<0?0:2;
    j = iXY2cIndex(eeDetId.ix());
    k = iXY2cIndex(eeDetId.iy());
  } else{
    throw cms::Exception("Not recognized subdetector. Probably a bug.");
  }
  int iDcc0 = ::dccIndex(i,j,k);
  assert(iDcc0>=0 && (unsigned)iDcc0<nDccs);
  return iDcc0+1;
}

EcalScDetId
EcalSelectiveReadoutValidation::superCrystalOf(const EEDetId& xtalId) const
{
  const int scEdge = 5;
  return EcalScDetId((xtalId.ix()-1)/scEdge+1,
		     (xtalId.iy()-1)/scEdge+1,
		     xtalId.zside());
}


EcalTrigTowerDetId
EcalSelectiveReadoutValidation::readOutUnitOf(const EBDetId& xtalId) const{
  return triggerTowerMap_->towerOf(xtalId);
}

EcalScDetId
EcalSelectiveReadoutValidation::readOutUnitOf(const EEDetId& xtalId) const{
  return superCrystalOf(xtalId);
}

