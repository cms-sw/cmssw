/*
 * \file EcalSelectiveReadoutValidation.cc
 *
 * $Date: 2007/05/21 13:21:52 $
 * $Revision: 1.2 $
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

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"
#include "CondFormats/DataRecord/interface/EcalTPParametersRcd.h"

#include <string.h>

using namespace cms;
using namespace edm;
using namespace std;

const double EcalSelectiveReadoutValidation::rad2deg = 45./atan(1.);

EcalSelectiveReadoutValidation::EcalSelectiveReadoutValidation(const ParameterSet& ps):
  ebDigis_(ps.getParameter<edm::InputTag>("EbDigiCollection"), false),
  eeDigis_(ps.getParameter<edm::InputTag>("EeDigiCollection"), false),
  ebNoZsDigis_(ps.getParameter<edm::InputTag>("EbUnsuppressedDigiCollection"),
	       false),
  eeNoZsDigis_(ps.getParameter<edm::InputTag>("EeUnsuppressedDigiCollection"),
	       false),
  ebSrFlags_(ps.getParameter<edm::InputTag>("EbSrFlagCollection"), false),
  eeSrFlags_(ps.getParameter<edm::InputTag>("EeSrFlagCollection"), false),
  ebSimHits_(ps.getParameter<edm::InputTag>("EbSimHitCollection"), false),
  eeSimHits_(ps.getParameter<edm::InputTag>("EeSimHitCollection"), false),
  tps_(ps.getParameter<edm::InputTag>("TrigPrimCollection"), false),
  ebRecHits_(ps.getParameter<edm::InputTag>("EbRecHitCollection"), false),
  eeRecHits_(ps.getParameter<edm::InputTag>("EeRecHitCollection"), false),
  triggerTowerMap_(0),
  tpParam_(0),
  localReco_(ps.getParameter<bool>("LocalReco")),
  weights_(ps.getParameter<vector<double> >("weights")),
  ievt_(0){
  
  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  
  if(outputFile_.size() != 0){
    LogInfo("OutputInfo") << " Ecal Digi Task histograms will be saved to '"
			  << outputFile_.c_str() << "'";
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

  //Data volume
  meDccVol_ = bookProfile("dccVol",
			  "DCC event fragment size;Dcc id; "
			  "<Event size> (kB)", nDccs, .5, .5+nDccs);
  
  meVolBLI_ = book1D("volBLI",
		     "Barrel low interest data volume;"
		     "Event size (kB);Nevts",
		     100, 0., 200.);
  
  meVolELI_ = book1D("volELI",
		     "Endcap low interest data volume;"
		     "Event size (kB);Nevts",
		     100, 0., 200.);
  
  meVolLI_ = book1D("volLI",
		    "ECAL low interest data volume;"
		    "Event size (kB);Nevts",
		    100, 0., 200.);
  
  meVolBHI_ = book1D("volBHI",
		     "Barrel high interest data volume;"
		     "Event size (kB);Nevts",
		     100, 0., 200.);
  
  meVolEHI_ = book1D("volEHI",
		     "Endcap high interest data volume;"
		     "Event size (kB);Nevts",
		     100, 0., 200.);
  
  meVolHI_ = book1D("volHI",
		    "ECAL high interest data volume;"
		    "Event size (kB);Nevts",
		    100, 0., 200.);
  
  meVolB_ = book1D("volB",
		   "Barrel data volume;Event size (kB);Nevts",
		   100, 0., 200.);
  
  meVolE_ = book1D("volE",
		   "Endcap data volume;Event size (kB);Nevts",
		   100, 0., 200.);
  
  meVol_ = book1D("vol",
		  "ECAL data volume;Event size (kB);Nevts",
		  100, 0., 200.);

  meChOcc_ = book2D("hChOcc",
		    "Crystal channel occupency after zero suppression;"
		    "iX0 / iEta0+120 / iX0 + 310;"
		    "iY0 / iPhi0 (starting from 0);"
		    "Event count",
		    410, -.5, 409.5,
		    360, -.5, 359.5);

  //TP
  meTp_ = book1D("tp",
		 "Trigger primitive TT E_{T};E_{T} (GeV);Event count",
		 100, 0., 10.);
  
  meTtf_ = book1D("ttFlag",
		  "Trigger primitive TT flag;Flag number;Event count",
		  8, -.5, 7.5);
  
  meTtfVsTp_ = book2D("ttfVsTp",
		      "Trigger tower flag vs TP;E_{T}(TT) (GeV);"
		      "Flag number",
		      100, 0., 10.,
		      8, -.5, 7.5);
  
  meTtfVsEtSum_ = book2D("ttfVsEtSum",
			 "Trigger tower flag vs #sumE_{T};"
			 "E_{T}(TT) (GeV);"
			 "TTF",
			 100, 0., 10.,
			 8, -.5, 7.5);
  
  meTpVsEtSum_ = book2D("tpVsEtSum",
			"Trigger primitive Et (TP) vs #sumE_{T};"
			"E_{T} (sum) (GeV);"
			"E_{T} (TP) (GeV)",
			100, 0., 10.,
			100, 0., 10.);
  
  const float ebMinE = 0.;
  const float ebMaxE = 120.;
 
  const float eeMinE = 0.;
  const float eeMaxE = 120.;
 
  const float ebMinNoise = -1.;
  const float ebMaxNoise = 1.;
 
  const float eeMinNoise = -1.;
  const float eeMaxNoise = 1.;

  const int evtMax = 500;
 
  meEbRecE_ = book1D("hEbRecE",
		     "Crystal reconstructed energy;E (GeV);Event count",
		     100, ebMinE, ebMaxE);

  meEbEMean_ = bookProfile("hEbEMean",
			   "EE <E_hit>;event #;<E_hit> (GeV)",
			   evtMax, .5, evtMax + .5);

  meEbNoise_ = book1D("hEbNoise",
		      "Crystal noise "
		      "(rec E of crystal without deposited energy)",
		      100, ebMinNoise, ebMaxNoise);

  meEbSimE_ = book1D("hEbSimE", "EB hit crystal simulated energy",
		     100, -1., 2.5);
 
  meEbRecEHitXtal_ = book1D("hEbRecEHitXtal",
			    "EB rec energy of hit crystals",
			    100, -1., 2.5);
 
  meEbRecVsSimE_ = book2D("hEbRecVsSimE",
			  "Crystal simulated vs reconstructed energy;"
			  "Esim (GeV);Erec GeV);Event count",
			  100, ebMinE, ebMaxE,
			  100, ebMinE, ebMaxE);
 
  meEbNoZsRecVsSimE_ = book2D("hEbNoZsRecVsSimE",
			      "Crystal no-zs simulated vs reconstructed "
			      "energy;"
			      "Esim (GeV);Erec GeV);Event count",
			      100, ebMinE, ebMaxE,
			      100, ebMinE, ebMaxE);
  
  meEeRecE_ = book1D("hEeRecE",
		     "EE crystal reconstructed energy;E (GeV);"
		     "Event count",
		     100, eeMinE, eeMaxE);
  
  meEeEMean_ = bookProfile("hEeEMean",
			   "EE <E_hit>;event #;<E_hit> (GeV)",
			   evtMax, .5, evtMax + .5);

  
  meEeNoise_ = book1D("hEeNoise",
		      "EE crystal noise "
		      "(rec E of crystal without deposited energy);"
		      "E (GeV); Event count",
		      200, eeMinNoise, eeMaxNoise);
  
  meEeSimE_ = book1D("hEeSimE", "EE hit crystal simulated energy",
		     100, -1., 2.5);
 
  meEeRecEHitXtal_ = book1D("hEeRecEHitXtal",
			    "EE rec energy of hit crystals",
			    100, -1., 2.5);
  
  meEeRecVsSimE_ = book2D("hEeRecVsSimE",
			  "EE crystal simulated vs reconstructed energy;"
			  "Esim (GeV);Erec GeV);Event count",
			  100, eeMinE, eeMaxE,
			  100, eeMinE, eeMaxE);
  
  meEeNoZsRecVsSimE_ = book2D("hEeNoZsRecVsSimE",
			      "EE crystal no-zs simulated vs "
			      "reconstructed "
			      "energy;Esim (GeV);Erec GeV);Event count",
			      100, eeMinE, eeMaxE,
			      100, eeMinE, eeMaxE);
}

void EcalSelectiveReadoutValidation::analyze(const Event& event,
					     const EventSetup& es){
  //retrieves event products:
  readAllCollections(event);

  //computes Et sum trigger tower crystals:
  setTtEtSums(es, *ebDigis_, *eeDigis_);
  
  //Data Volume
  analyzeDataVolume(event, es);
  
  //EB digis
  analyzeEB(event, es);
  
  //EE digis
  analyzeEE(event, es);
  
  //TP
  analyzeTP(event, es);
}


void EcalSelectiveReadoutValidation::analyzeEE(const edm::Event& event,
					       const edm::EventSetup& es){
  for(int iZ0=0; iZ0<nEndcaps; ++iZ0){
    for(int iX0=0; iX0<nEeX; ++iX0){
      for(int iY0=0; iY0<nEeY; ++iY0){
        eeEnergies[iZ0][iX0][iY0].noZsRecE = -numeric_limits<double>::max();
        eeEnergies[iZ0][iX0][iY0].recE = -numeric_limits<double>::max();
        eeEnergies[iZ0][iX0][iY0].simE = 0; //must be set to zero.
        eeEnergies[iZ0][iX0][iY0].simHit = 0; 
      }
    }
  }
  
  // gets the endcap geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry_p
    = (*geoHandle).getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  CaloSubdetectorGeometry const& geometry = *geometry_p;

  //EE unsupressed digis:
  for(EEDigiCollection::const_iterator it = eeNoZsDigis_->begin();
      it != eeNoZsDigis_->end(); ++it){
    const EEDataFrame& frame = *it;
    int iX0 = iXY2cIndex(static_cast<const EEDetId&>(frame.id()).ix());
    int iY0 = iXY2cIndex(static_cast<const EEDetId&>(frame.id()).iy());
    int iZ0 = static_cast<const EEDetId&>(frame.id()).zside()>0?1:0;

    if(iX0<0 || iX0>=nEeX){
      cout << "iX0 (= " << iX0 << ") is out of range ("
           << "[0," << nEeX -1 << "]\n";
    }
    if(iY0<0 || iY0>=nEeY){
      cout << "iY0 (= " << iY0 << ") is out of range ("
           << "[0," << nEeY -1 << "]\n";
    }
    //    cout << "EE no ZS energy computation..." ;
    eeEnergies[iZ0][iX0][iY0].noZsRecE = frame2Energy(frame);

    const GlobalPoint xtalPos
      = geometry.getGeometry(frame.id())->getPosition();
    
    eeEnergies[iZ0][iX0][iY0].phi = rad2deg*((double)xtalPos.phi());
    eeEnergies[iZ0][iX0][iY0].eta = xtalPos.eta();
  }

  //EE rec hits:
  if(!localReco_){
    for(EcalRecHitCollection::const_iterator it
          = eeRecHits_->begin();
        it != eeRecHits_->end(); ++it){
      const EcalRecHit& hit = *it;
      int iX0 = iXY2cIndex(static_cast<const EEDetId&>(hit.id()).ix());
      int iY0 = iXY2cIndex(static_cast<const EEDetId&>(hit.id()).iy());
      int iZ0 = static_cast<const EEDetId&>(hit.id()).zside()>0?1:0;
      
      if(iX0<0 || iX0>=nEeX){
        cout << "iX0 (= " << iX0 << ") is out of range ("
             << "[0," << nEeX -1 << "]\n";
      }
      if(iY0<0 || iY0>=nEeY){
        cout << "iY0 (= " << iY0 << ") is out of range ("
             << "[0," << nEeY -1 << "]\n";
      }
      //    cout << "EE no ZS energy computation..." ;
      eeEnergies[iZ0][iX0][iY0].recE = hit.energy();
    }
  }

  //EE sim hits:
  for(vector<PCaloHit>::const_iterator it = eeSimHits_->begin();
      it != eeSimHits_->end(); ++it){
    const PCaloHit& simHit = *it;
    EEDetId detId(simHit.id());
    int iX = detId.ix();
    int iX0 =iXY2cIndex(iX);
    int iY = detId.iy();
    int iY0 = iXY2cIndex(iY);
    int iZ0 = detId.zside()>0?1:0;
    eeEnergies[iZ0][iX0][iY0].simE += simHit.energy();
    ++eeEnergies[iZ0][iX0][iY0].simHit;
  }

  //EE suppressed digis
  for(EEDigiCollection::const_iterator it = eeDigis_->begin();
      it != eeDigis_->end(); ++it){
    const EEDataFrame& frame = *it;
    int iX0 = iXY2cIndex(static_cast<const EEDetId&>(frame.id()).ix());
    int iY0 = iXY2cIndex(static_cast<const EEDetId&>(frame.id()).iy());
    int iZ0 = static_cast<const EEDetId&>(frame.id()).zside()>0?1:0;
    if(iX0<0 || iX0>=nEeX){
        cout << "iX0 (= " << iX0 << ") is out of range ("
             << "[0," << nEeX -1 << "]\n";
    }
    if(iY0<0 || iY0>=nEeY){
        cout << "iY0 (= " << iY0 << ") is out of range ("
             << "[0," << nEeY -1 << "]\n";
    }
    //    cout << "EE zs Energy computation...";
      if(localReco_){
        eeEnergies[iZ0][iX0][iY0].recE = frame2Energy(frame);
      }
      meChOcc_->Fill(iX0 + iZ0*310, iY0);
    } //next ZS digi.
  
  for(int iZ0=0; iZ0<nEndcaps; ++iZ0){
    for(int iX0=0; iX0<nEeX; ++iX0){
      for(int iY0=0; iY0<nEeY; ++iY0){        
        double recE = eeEnergies[iZ0][iX0][iY0].recE;
        if(recE==-numeric_limits<double>::max()) continue; //not a crystal or ZS
        meEeRecE_->Fill(eeEnergies[iZ0][iX0][iY0].recE);
	
        meEeEMean_->Fill(ievt_+1,
			 eeEnergies[iZ0][iX0][iY0].recE);
        
        if(!eeEnergies[iZ0][iX0][iY0].simHit){//noise only crystal channel
          meEeNoise_->Fill(eeEnergies[iZ0][iX0][iY0].noZsRecE);
        } else{
          meEeSimE_->Fill(eeEnergies[iZ0][iX0][iY0].simE);	  
          meEeRecEHitXtal_->Fill(eeEnergies[iZ0][iX0][iY0].recE);
        }
	meEeRecVsSimE_->Fill(eeEnergies[iZ0][iX0][iY0].simE,
			     eeEnergies[iZ0][iX0][iY0].recE);
	meEeNoZsRecVsSimE_->Fill(eeEnergies[iZ0][iX0][iY0].simE,
				 eeEnergies[iZ0][iX0][iY0].noZsRecE); 
      }
    }
  }
} //end of analyzeEE

void
EcalSelectiveReadoutValidation::analyzeEB(const edm::Event& event,
					  const edm::EventSetup& es){  
  vector<pair<int,int> > xtalEtaPhi;
  xtalEtaPhi.reserve(nEbPhi*nEbEta);
  for(int iEta0=0; iEta0<nEbEta; ++iEta0){
    for(int iPhi0=0; iPhi0<nEbPhi; ++iPhi0){
      ebEnergies[iEta0][iPhi0].noZsRecE = -numeric_limits<double>::max();
      ebEnergies[iEta0][iPhi0].recE = -numeric_limits<double>::max();
      ebEnergies[iEta0][iPhi0].simE = 0; //must be zero.
      ebEnergies[iEta0][iPhi0].simHit = 0;
      xtalEtaPhi.push_back(pair<int,int>(iEta0, iPhi0));
    }
  }
  
  // get the barrel geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry_p
    = (*geoHandle).getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  CaloSubdetectorGeometry const& geometry = *geometry_p;

  //EB unsuppressed digis:
  for(EBDigiCollection::const_iterator it = ebDigis_->begin();
      it != ebDigis_->end(); ++it){
    const EBDataFrame& frame = *it;
    int iEta0 = iEta2cIndex(static_cast<const EBDetId&>(frame.id()).ieta());
    int iPhi0 = iPhi2cIndex(static_cast<const EBDetId&>(frame.id()).iphi());
    if(iEta0<0 || iEta0>=nEbEta){
      stringstream s;
      s << "EcalSelectiveReadoutValidation: "
	<< "iEta0 (= " << iEta0 << ") is out of range ("
	<< "[0," << nEbEta -1 << "]\n";
      throw cms::Exception(s.str());
    }
    if(iPhi0<0 || iPhi0>=nEbPhi){
      stringstream s;
      s << "EcalSelectiveReadoutValidation: "
	<< "iPhi0 (= " << iPhi0 << ") is out of range ("
	<< "[0," << nEbPhi -1 << "]\n";
      throw cms::Exception(s.str());
    }
    
    ebEnergies[iEta0][iPhi0].noZsRecE = frame2Energy(frame);

    const GlobalPoint xtalPos
      = geometry.getGeometry(frame.id())->getPosition();

    ebEnergies[iEta0][iPhi0].phi = rad2deg*((double)xtalPos.phi());
    ebEnergies[iEta0][iPhi0].eta = xtalPos.eta();
  } //next non-zs digi

  //EB sim hits
  for(vector<PCaloHit>::const_iterator it = ebSimHits_->begin();
      it != ebSimHits_->end(); ++it){
    const PCaloHit& simHit = *it;
    EBDetId detId(simHit.id());
    int iEta = detId.ieta();
    int iEta0 =iEta2cIndex(iEta);
    int iPhi = detId.iphi();
    int iPhi0 = iPhi2cIndex(iPhi);
    ebEnergies[iEta0][iPhi0].simE += simHit.energy();
    ++ebEnergies[iEta0][iPhi0].simHit;
  }
  
  bool crystalShot[nEbEta][nEbPhi];
  for(int iEta0=0; iEta0<nEbEta; ++iEta0){
    for(int iPhi0=0; iPhi0<nEbPhi; ++iPhi0){
      crystalShot[iEta0][iPhi0] = false;
    }
  }
  
  int nEbDigi = 0;

  for(EBDigiCollection::const_iterator it = ebDigis_->begin();
      it != ebDigis_->end(); ++it){
    ++nEbDigi;
    const EBDataFrame& frame = *it;
    int iEta = static_cast<const EBDetId&>(frame.id()).ieta();
      int iPhi = static_cast<const EBDetId&>(frame.id()).iphi();
      int iEta0 = iEta2cIndex(iEta);
      int iPhi0 = iPhi2cIndex(iPhi);
      if(iEta0<0 || iEta0>=nEbEta){
	throw (cms::Exception("EcalSelectiveReadoutValidation")
	       << "iEta0 (= " << iEta0 << ") is out of range ("
	       << "[0," << nEbEta -1 << "]");
      }
      if(iPhi0<0 || iPhi0>=nEbPhi){
	throw (cms::Exception("EcalSelectiveReadoutValidation")
	       << "iPhi0 (= " << iPhi0 << ") is out of range ("
	       << "[0," << nEbPhi -1 << "]");
      }
      assert(iEta0>=0 && iEta0<nEbEta);
      assert(iPhi0>=0 && iPhi0<nEbPhi);
      if(!crystalShot[iEta0][iPhi0]){
        crystalShot[iEta0][iPhi0] = true;
      } else{
        cout << "Error: several digi for same crystal!";
        abort();
      }
      if(localReco_){
        ebEnergies[iEta0][iPhi0].recE = frame2Energy(frame);
      }
      meChOcc_->Fill(iEta0+120, iPhi0);
  } //next EB digi

  if(!localReco_){
    for(EcalRecHitCollection::const_iterator it
	  = ebRecHits_->begin();
        it != ebRecHits_->end(); ++it){
      ++nEbDigi;
      const EcalRecHit& hit = *it;
      int iEta = static_cast<const EBDetId&>(hit.id()).ieta();
      int iPhi = static_cast<const EBDetId&>(hit.id()).iphi();
      int iEta0 = iEta2cIndex(iEta);
      int iPhi0 = iPhi2cIndex(iPhi);
      if(iEta0<0 || iEta0>=nEbEta){
        cout << "iEta0 (= " << iEta0 << ") is out of range ("
             << "[0," << nEbEta -1 << "]\n";
      }
      if(iPhi0<0 || iPhi0>=nEbPhi){
        cout << "iPhi0 (= " << iPhi0 << ") is out of range ("
             << "[0," << nEbPhi -1 << "]\n";
      }
      ebEnergies[iEta0][iPhi0].recE = hit.energy();
    }
  }

  //sorts crystal in increasing sim hit energy. ebEnergies[][].simE
  //must be set beforehand:
  sort(xtalEtaPhi.begin(), xtalEtaPhi.end(), Sorter(this));
  
  //   cout << "\niEta\tiPhi\tsimE\tnoZsE\tzsE\n";
  for(unsigned int i=0; i<xtalEtaPhi.size(); ++i){
    int iEta0 = xtalEtaPhi[i].first;
    int iPhi0=  xtalEtaPhi[i].second;
    energiesEb_t& energies = ebEnergies[iEta0][iPhi0];

    double recE = ebEnergies[iEta0][iPhi0].recE;
    if(recE!=-numeric_limits<double>::max()){//not zero suppressed
      meEbRecE_->Fill(ebEnergies[iEta0][iPhi0].recE);
      meEbEMean_->Fill(ievt_+1, recE);
    } //not zero suppressed
        
    if(!energies.simHit){//noise only crystal channel
      meEbNoise_->Fill(energies.noZsRecE);
    } else{
      meEbSimE_->Fill(energies.simE);
      meEbRecEHitXtal_->Fill(energies.recE);
    }
    meEbRecVsSimE_->Fill(energies.simE, energies.recE);
    meEbNoZsRecVsSimE_->Fill(energies.simE, energies.noZsRecE);
  }
}

EcalSelectiveReadoutValidation::~EcalSelectiveReadoutValidation(){ 
  if(outputFile_.size()!=0) dbe_->save(outputFile_);
}

void EcalSelectiveReadoutValidation::beginJob(const EventSetup& setup){
  // endcap mapping
  edm::ESHandle<EcalTrigTowerConstituentsMap> hTriggerTowerMap;
  setup.get<IdealGeometryRecord>().get(hTriggerTowerMap);
  triggerTowerMap_ = hTriggerTowerMap.product();

  //electronics map
  ESHandle< EcalElectronicsMapping > ecalmapping;
  setup.get< EcalMappingRcd >().get(ecalmapping);
  elecMap_ = ecalmapping.product();
  
  //trigger primitive parameters:
  edm::ESHandle<EcalTPParameters> hTpParam;
  setup.get<EcalTPParametersRcd>().get(hTpParam);
  tpParam_=hTpParam.product();
}

void EcalSelectiveReadoutValidation::endJob(){
}

void
EcalSelectiveReadoutValidation::analyzeTP(const edm::Event& event,
					  const edm::EventSetup& es){  
  for(EcalTrigPrimDigiCollection::const_iterator it = tps_->begin();
      it != tps_->end(); ++it){
    const int iTcc = elecMap_->TCCid(it->id());
    const int iTT = elecMap_->iTT(it->id());
    const double tpEt = tpParam_->getTPGinGeVEB(iTcc, iTT,
						it->compressedEt());
        
    const int iEta0 = iTTEta2cIndex(it->id().ieta());
    const int iPhi0 = iTTEta2cIndex(it->id().iphi());
    const double etSum = ttEtSums[iEta0][iPhi0];
    meTp_->Fill(tpEt);
    meTpVsEtSum_->Fill(etSum, tpEt);
    meTtf_->Fill(it->ttFlag());
    meTtfVsTp_->Fill(tpEt, it->ttFlag());
    meTtfVsEtSum_->Fill(etSum, it->ttFlag());
  }
}

void EcalSelectiveReadoutValidation::analyzeDataVolume(const Event& e,
						       const EventSetup& es){
  
  anaDigiInit();

  //Barrel
  for(std::vector<EBDataFrame>::const_iterator it = ebDigis_->begin() ;
      it != ebDigis_->end();
      ++it){
    anaDigi(*it, *ebSrFlags_);
  }

  // Endcap
  for(std::vector<EEDataFrame>::const_iterator it = eeDigis_->begin() ;
      it != eeDigis_->end() ;
	++it){
    anaDigi(*it, *eeSrFlags_);
  }

  //histos
  for(unsigned iDcc = 0; iDcc <  nDccs; ++iDcc){ 
    meDccVol_->Fill(iDcc, getDccEventSize(iDcc, nPerDcc_[iDcc])/kByte_);
  }


  //low interesest channels:
  double a = getEbEventSize(nEbLI_)/kByte_;
  meVolBLI_->Fill(a);
  double b = getEeEventSize(nEeLI_)/kByte_;
  meVolELI_->Fill(b);	
  meVolLI_->Fill(a+b);	

  //high interest chanels:
  a = getEbEventSize(nEbHI_)/kByte_;
  meVolBHI_->Fill(a);
  b = getEeEventSize(nEeHI_)/kByte_;
  meVolEHI_->Fill(b);	
  meVolHI_->Fill(a+b);

  //any-interest channels:
  a = getEbEventSize(nEb_)/kByte_;
  meVolB_->Fill(a);
  b = getEeEventSize(nEe_)/kByte_;
  meVolE_->Fill(b);
  meVol_->Fill(a+b);
  ++ievt_;
}


template<class T, class U>
void EcalSelectiveReadoutValidation::anaDigi(const T& frame,
					     const U& srFlagColl){
  const DetId& xtalId = frame.id();
  typename U::const_iterator srf = srFlagColl.find(readOutUnitOf(frame.id()));
  
  if(srf == srFlagColl.end()){
    throw cms::Exception("EcalSelectiveReadoutValidation")
      << __FILE__ << ":" << __LINE__ << ": SR flag not found";
  }
  
  bool highInterest = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK)
		       == EcalSrFlag::SRF_FULL);
  
  bool barrel = (xtalId.subdetId()==EcalBarrel);

  if(barrel){
    ++nEb_;
    if(highInterest){
      ++nEbHI_;
    } else{//low interest
      ++nEbLI_;
    }
  } else{//endcap
    ++nEe_;
    if(highInterest){
      ++nEeHI_;
    } else{//low interest
      ++nEeLI_;
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

double EcalSelectiveReadoutValidation::frame2Energy(const EcalDataFrame& frame) const{
  static bool firstCall = true;
  if(firstCall){
    cout << "Weights:";
    for(unsigned i=0; i<weights_.size();++i){
      cout << "\t" << weights_[i];
    }
    cout << "\n";
    firstCall = false;
  }
  double adc2GeV = 0.;
  
  if(typeid(EBDataFrame)==typeid(frame)){//barrel APD
    adc2GeV = .035;
  } else if(typeid(EEDataFrame)==typeid(frame)){//endcap VPT
    adc2GeV = 0.06;
  } else{
    assert(false);
  }
  
  double acc = 0;
  
  const int n = min(frame.size(), (int)weights_.size());

  double gainInv[] = {12., 1., 6., 12.};

  for(int i=0; i < n; ++i){
    acc += weights_[i]*frame[i].adc()*gainInv[frame[i].gainId()]*adc2GeV;
  }
  return acc;
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
    throw cms::Exception("EcalSelectiveReadoutValidation")
      <<"Not recognized subdetector. Probably a bug.";
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

void
EcalSelectiveReadoutValidation::setTtEtSums(const edm::EventSetup& es,
					    const EBDigiCollection& ebDigis,
					    const EEDigiCollection& eeDigis){
  //ecal geometry:
  static const CaloSubdetectorGeometry* eeGeometry = 0;
  static const CaloSubdetectorGeometry* ebGeometry = 0;
  if(eeGeometry==0 || ebGeometry==0){
    edm::ESHandle<CaloGeometry> geoHandle;
    es.get<IdealGeometryRecord>().get(geoHandle);
    eeGeometry
      = (*geoHandle).getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    ebGeometry
      = (*geoHandle).getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  }
  
  //init etSum array:
  for(int iEta0 = 0; iEta0 < nTtEta; ++iEta0){
    for(int iPhi0 = 0; iPhi0 < nTtPhi; ++iPhi0){
      ttEtSums[iEta0][iPhi0] = 0.;
    }
  }
  
  for(EBDigiCollection::const_iterator it = ebDigis_->begin();
      it != ebDigis_->end(); ++it){
    const EBDataFrame& frame = *it;
    const EcalTrigTowerDetId& ttId = triggerTowerMap_->towerOf(frame.id());
    //      LogDebug("TT")
    //        <<  ((EBDetId&)frame.id()).ieta()
    //        << "," << ((EBDetId&)frame.id()).iphi()
    //        << " -> " << ttId.ieta() << "," << ttId.iphi();
    const int iTTEta0 = iTTEta2cIndex(ttId.ieta());
    const int iTTPhi0 = iTTPhi2cIndex(ttId.iphi());
    double theta = ebGeometry->getGeometry(frame.id())->getPosition().theta();
    double e = frame2EnergyForTp(frame);
    if((frame2EnergyForTp(frame,-1) < e) && (frame2EnergyForTp(frame, 1) < e)){
      ttEtSums[iTTEta0][iTTPhi0] += e*sin(theta);
    }
  }
  
  for(EEDigiCollection::const_iterator it = eeDigis.begin();
      it != eeDigis.end(); ++it){
    const EEDataFrame& frame = *it;
    const EcalTrigTowerDetId& ttId = triggerTowerMap_->towerOf(frame.id());
    const int iTTEta0 = iTTEta2cIndex(ttId.ieta());
    const int iTTPhi0 = iTTPhi2cIndex(ttId.iphi());
    //     LogDebug("TT") << ": EE xtal->TT "
    //        <<  ((EEDetId&)frame.id()).ix()
    //        << "," << ((EEDetId&)frame.id()).iy()
    //        << " -> " << ttId.ieta() << "," << ttId.iphi() << "\n";
    double theta = eeGeometry->getGeometry(frame.id())->getPosition().theta();
    double e = frame2EnergyForTp(frame);
    if((frame2EnergyForTp(frame,-1) < e) && (frame2EnergyForTp(frame, 1) < e)){
      ttEtSums[iTTEta0][iTTPhi0] += e*sin(theta);
    }
  }
  
  //dealing with pseudo-TT in two inner EE eta-ring:
  int innerTTEtas[] = {0, 1, 54, 55};
  for(unsigned iRing = 0; iRing < sizeof(innerTTEtas)/sizeof(innerTTEtas[0]);
      ++iRing){
    int iTTEta0 = innerTTEtas[iRing];
    //this detector eta-section is divided in only 36 phi bins
    //For this eta regions,
    //current tower eta numbering scheme is inconsistent. For geometry
    //version 133:
    //- TT are numbered from 0 to 72 for 36 bins
    //- some TT have an even index, some an odd index
    //For geometry version 125, there are 72 phi bins.
    //The code below should handle both geometry definition.
    //If there are 72 input trigger primitives for each inner eta-ring,
    //then the average of the trigger primitive of the two pseudo-TT of
    //a pair (nEta, nEta+1) is taken as Et of both pseudo TTs.
    //If there are only 36 input TTs for each inner eta ring, then half
    //of the present primitive of a pseudo TT pair is used as Et of both
    //pseudo TTs.
    
    for(unsigned iTTPhi0 = 0; iTTPhi0 < nTtPhi-1; iTTPhi0 += 2){
      double et = .5*(ttEtSums[iTTEta0][iTTPhi0]
		      +ttEtSums[iTTEta0][iTTPhi0+1]);
      //divides the TT into 2 phi bins in order to match with 72 phi-bins SRP
      //scheme or average the Et on the two pseudo TTs if the TT is already
      //divided into two trigger primitives.
      ttEtSums[iTTEta0][iTTPhi0] = et;
      ttEtSums[iTTEta0][iTTPhi0+1] = et;
    }
  }
}

template<class T>
double EcalSelectiveReadoutValidation::frame2EnergyForTp(const T& frame,
							 int offset) const{
  //we have to start by 0 in order to handle offset=-1
  //(however Fenix FIR has AFAK only 5 taps)
  double weights[] = {0., -1/3., -1/3., -1/3., 0., 1.};
  
  double adc2GeV = 0.;
  if(typeid(frame) == typeid(EBDataFrame)){
    adc2GeV = 0.035;
  } else if(typeid(frame) == typeid(EEDataFrame)){
    adc2GeV = 0.060;
  } else{ //T is an invalid type!
    //TODO: replace message by a cms exception
    throw cms::Exception("Severe Error")
      << __FILE__ << ":" << __LINE__ << ": "
      << "this is a bug. Please report it.\n";
  }
  
  double acc = 0;
  
  const int n = min<int>(frame.size(), sizeof(weights)/sizeof(weights[0]));
  
  double gainInv[] = {12., 1., 6., 12}; 

  for(int i=offset; i < n; ++i){
    int iframe = i + offset;
    if(iframe>=0 && iframe<frame.size()){
      acc += weights[i]*frame[iframe].adc()
	*gainInv[frame[iframe].gainId()]*adc2GeV;
      //cout << (iframe>offset?"+":"")
      //     << frame[iframe].adc() << "*" << gainInv[frame[iframe].gainId()]
      //     << "*" << adc2GeV << "*(" << weights[i] << ")";
    }
  }
  //cout << "\n";
  return acc;
}

MonitorElement* EcalSelectiveReadoutValidation::book1D(const std::string& name, const std::string& title, int nbins, double xmin, double xmax){
  MonitorElement* result = dbe_->book1D(name, title, nbins, xmin, xmax);
  if(result==0){
    throw cms::Exception("Histo")
      << "Failed to book histogram " << name;
  }
  return result;
}

MonitorElement* EcalSelectiveReadoutValidation::book2D(const std::string& name, const std::string& title, int nxbins, double xmin, double xmax, int nybins, double ymin, double ymax){
  MonitorElement* result = dbe_->book2D(name, title, nxbins, xmin, xmax,
					nybins, ymin, ymax);
  if(result==0){
    throw cms::Exception("Histo")
      << "Failed to book histogram " << name;
  }
  return result;
}

MonitorElement* EcalSelectiveReadoutValidation::bookProfile(const std::string& name, const std::string& title, int nbins, double xmin, double xmax){
  MonitorElement* result = dbe_->bookProfile(name, title, nbins, xmin, xmax,
					     0, 0, 0);
  if(result==0){
    throw cms::Exception("Histo")
      << "Failed to book histogram " << name;
  }
  return result;
}

void EcalSelectiveReadoutValidation::readAllCollections(const edm::Event& event){
  ebRecHits_.read(event);
  eeRecHits_.read(event);
  ebDigis_.read(event);
  eeDigis_.read(event);
  ebNoZsDigis_.read(event);
  eeNoZsDigis_.read(event);
  ebSrFlags_.read(event);
  eeSrFlags_.read(event);
  ebSimHits_.read(event);
  eeSimHits_.read(event);
  tps_.read(event);
}
