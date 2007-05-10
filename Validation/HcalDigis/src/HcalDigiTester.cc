#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Validation/HcalDigis/interface/HcalDigiTester.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>


template<class Digi>
void HcalDigiTester::reco(const edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  double fphi_mc = -999999.;  // phi of initial particle from HepMC
  double feta_mc = -999999.;  // eta of initial particle from HepMC
  int inumTower = 0; // number of towers where sum 4567(4-HF) greater 10;
  bool MC=false;

  edm::Handle<edm::HepMCProduct> evtMC;
  try{
    //  iEvent.getByLabel("VtxSmeared",evtMC);
    iEvent.getByLabel("source",evtMC);
    MC=true;
    //    std::cout << "source HepMCProduct found"<< std::endl;
  }
  catch( char * str ) {
    MC=false;
    std::cout << "no HepMCProduct found"<<str<< std::endl;    
  }
  
  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evtMC->GetEvent()));
  for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	p != myGenEvent->particles_end(); ++p ) {
    fphi_mc = (*p)->momentum().phi();
    feta_mc = (*p)->momentum().eta();
  }

 
  typename   edm::Handle<edm::SortedCollection<Digi> > hbhe;
  typename edm::SortedCollection<Digi>::const_iterator ihbhe;
  
  ievent++;
   
  // ADC2fC 

  const HcalQIEShape* shape = conditions->getHcalShape();
  HcalCalibrations calibrations;
  CaloSamples tool;

  // loop over the digis
  int ndigis=0;
  
  float sumamplfC = 0;
  iEvent.getByType (hbhe) ;

  int subdet = 1;
  if (hcalselector_ == "HB"  ) subdet = 1;
  if (hcalselector_ == "HE"  ) subdet = 2;
  if (hcalselector_ == "HO"  ) subdet = 3;
  if (hcalselector_ == "HF"  ) subdet = 4; 

  for (ihbhe=hbhe->begin();ihbhe!=hbhe->end();ihbhe++)
    {

      HcalDetId cell(ihbhe->id()); 
      
      if (cell.subdet()== subdet   ) 
     	{
	  const CaloCellGeometry* cellGeometry =
	    geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	  double fEta = cellGeometry->getPosition ().eta () ;
	  double fPhi = cellGeometry->getPosition ().phi () ;

	      
	  conditions->makeHcalCalibration(cell, &calibrations);
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
	  HcalCoderDb coder (*channelCoder, *shape);
	  coder.adc2fC(*ihbhe,tool);
	      
	  float amplRecHitfC = 0;
	  float amplfC = 0;

	
	  for  (int ii=0;ii<tool.size();ii++)
	    {
	      int capid = (*ihbhe)[ii].capid();
	      amplfC+=(tool[ii]-calibrations.pedestal(capid));
	      if ( fabs(feta_mc-fEta) < 0.087/2. && acos(cos(fphi_mc-fPhi))<0.087/2.  ) 
		                            monitor()->fillTimeSlice( ii , tool[ii]);
 	       
	      if (hcalselector_ != "HF" && ii>=4 && ii<=7)
		{
		  amplRecHitfC+=(tool[ii]-calibrations.pedestal(capid));	   
		  monitor()->fillPedestalfC(calibrations.pedestal(capid));
		  monitor()->fillDigiMinusPedfC(tool[ii]-calibrations.pedestal(capid));
		}
	      if (hcalselector_ == "HF" && ii==3 && ii)
		{
		  amplRecHitfC+=(tool[3]-calibrations.pedestal(capid));
		  monitor()->fillPedestalfC(calibrations.pedestal(capid));
		  monitor()->fillDigiMinusPedfC(tool[3]-calibrations.pedestal(capid));
		}
	    }

	  monitor()->fillADC0count((*ihbhe)[0].adc());
	  monitor()->fillADC0fC(tool[0]);
	  
	  if (amplRecHitfC>10.)
	    {
	      sumamplfC += amplRecHitfC;	  
	      inumTower++;  // count towers with sum in main bins greater that 10 fC 
	      monitor()->fillEta(fEta);  // eta of tower 
	      monitor()->fillPhi(fPhi);  // phi of tower
	    }

	  monitor()->fillPhiMC(fphi_mc);
	  monitor()->fillEtaMC(feta_mc);
	  ndigis++;

	  if(amplfC > 50.)
	    {
	      // now do a few selected individual bins, if it's big enough
	      float fBin5  = tool[4];
	      float fBin67 = tool[5] + tool[6];


	      fBin5 -= calibrations.pedestal((*ihbhe)[4].capid());
	  
	      fBin67 -= (calibrations.pedestal((*ihbhe)[5].capid())
			 + calibrations.pedestal((*ihbhe)[6].capid()));
      
	      //fBin12 is a pedestal, others are percentages
	      if(amplfC > 0)
		{
		  fBin5 /= amplfC;
		  fBin67 /= amplfC;
		}

	      monitor()->fillBin5Frac(fBin5);
	      monitor()->fillBin67Frac(fBin67);
	    }
	
	}
    }
  monitor()->fillBin4567Frac(sumamplfC);
  monitor()->fillNTowersGt10(inumTower);

  edm::Handle<edm::PCaloHitContainer> hcalHits ;
  // iEvent.getByLabel("SimG4Object","HcalHits",hcalHits);
  iEvent.getByLabel("g4SimHits","HcalHits",hcalHits);
      
  const edm::PCaloHitContainer * simhitResult = hcalHits.product () ;
      
  float fEnergySimHits = 0; 
  for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin () ;
       simhits != simhitResult->end () ;
       ++simhits)
    {    
      HcalDetId detId(simhits->id());
      //  1 == HB
      if (detId.subdet()== subdet  ){  
	fEnergySimHits += simhits->energy(); 
      }
    }

  monitor()->fillDigiSimhit(fEnergySimHits, sumamplfC);
  monitor()->fillRatioDigiSimhit(sumamplfC/fEnergySimHits);
  monitor()->fillDigiSimhitProfile(fEnergySimHits, sumamplfC);
  monitor()->fillSumDigis(sumamplfC);
  monitor()->fillNDigis(ndigis);

}



HcalDigiTester::HcalDigiTester(const edm::ParameterSet& iConfig)
  : dbe_(edm::Service<DaqMonitorBEInterface>().operator->()),
    outputFile_(iConfig.getUntrackedParameter<std::string>("outputFile", "")),
    hcalselector_(iConfig.getUntrackedParameter<std::string>("hcalselector", "all")),
    subpedvalue_(iConfig.getUntrackedParameter<bool>("subpedvalue", "true")),
    monitors_()
{
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal Digi Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal Digi Task histograms will NOT be saved";
  }


  if ( dbe_ ) {
    dbe_->setCurrentFolder("HcalDigiTask");
  }

}
   

HcalDigiTester::~HcalDigiTester()
{
  std::cout << " outputFile_.size() =  " << outputFile_.size() << std::endl;
  std::cout << " dbe_ = " << dbe_ << std::endl; 
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}

void HcalDigiTester::endJob() {
  std::cout << " outputFile_.size() =  " << outputFile_.size() << std::endl;
  std::cout << " dbe_ = " << dbe_ << std::endl; 
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void HcalDigiTester::beginJob(const edm::EventSetup& c){

}


HcalSubdetDigiMonitor * HcalDigiTester::monitor()
{
  std::map<std::string, HcalSubdetDigiMonitor*>::iterator monitorItr
    = monitors_.find(hcalselector_);

  if(monitorItr == monitors_.end())
    {
      HcalSubdetDigiMonitor* m = new HcalSubdetDigiMonitor(dbe_, hcalselector_);
      std::pair<std::string, HcalSubdetDigiMonitor*> mapElement(
								hcalselector_, m);
      monitorItr = monitors_.insert(mapElement).first;
    }
  return monitorItr->second;
}

void
HcalDigiTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  iSetup.get<IdealGeometryRecord>().get (geometry);
  iSetup.get<HcalDbRecord>().get(conditions);

  if (hcalselector_ == "HB" ) reco<HBHEDataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HE" ) reco<HBHEDataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HO" ) reco<HODataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HF" ) reco<HFDataFrame>(iEvent,iSetup);  
                                                          
  if (hcalselector_ == "noise") 
    {
      hcalselector_ = "HB";
      reco<HBHEDataFrame>(iEvent,iSetup);
      hcalselector_ = "HE";
      reco<HBHEDataFrame>(iEvent,iSetup);
      hcalselector_ = "HO";
      reco<HODataFrame>(iEvent,iSetup);
      hcalselector_ = "HF";
      reco<HFDataFrame>(iEvent,iSetup);
      hcalselector_ = "noise";
    }

}

