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

#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>

using namespace cms;
using namespace edm;
using namespace std;




template<class Digi>
void HcalDigiTester::reco(const edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
 
  typename   edm::Handle<edm::SortedCollection<Digi> > hbhe;
  typename edm::SortedCollection<Digi>::const_iterator ihbhe;
  using namespace edm;

 

  
  
  // ADC2fC 


  const HcalQIEShape* shape = conditions->getHcalShape();

  HcalCalibrations calibrations;
 
  CaloSamples tool;

  // loop over the digis
  int ndigis=0;
  
  float fAdcSum = 0;// sum of all ADC counts in terms of fC
  iEvent.getByType (hbhe) ;

  int subdet = 1;
  
  if (hcalselector_ == "HB"  ) subdet = 1;
  if (hcalselector_ == "HE"  ) subdet = 2;
  if (hcalselector_ == "HO"  ) subdet = 3;
  if (hcalselector_ == "HF"  ) subdet = 4; 

  for (ihbhe=hbhe->begin();ihbhe!=hbhe->end();ihbhe++)
	{
	  HcalDetId cell(ihbhe->id()); 
	  if (cell.subdet()== subdet  ) 
	    {
	      const CaloCellGeometry* cellGeometry =
          geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	      double fEta = cellGeometry->getPosition ().eta () ;
	      double fPhi = cellGeometry->getPosition ().phi () ;
        monitor()->fillEta(fEta);
        monitor()->fillPhi(fPhi);
	      
	      conditions->makeHcalCalibration(cell, &calibrations);
	      const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
	      HcalCoderDb coder (*channelCoder, *shape);
	      coder.adc2fC(*ihbhe,tool);
	      
        float fDigiSum = 0;
	      for  (int ii=0;ii<tool.size();ii++)
        {
          int capid = (*ihbhe)[ii].capid();
          if (subpedvalue_) fDigiSum+=(tool[ii]-calibrations.pedestal(capid));
          if (!subpedvalue_) fDigiSum+=(tool[ii] - pedvalue);
        }
        fAdcSum += fDigiSum;

        monitor()->fillPedestal((*ihbhe)[0].adc());
        monitor()->fillPedestal((*ihbhe)[1].adc());
 
        if(fDigiSum > 50.)
        {
//std::cout << (*ihbhe) << std::endl;
//std::cout << tool << std::endl;
//std::cout << pedvalue << std::endl;
          // now do a few selected individual bins, if it's big enough
          float fBin5  = tool[4];
          float fBin67 = tool[5] + tool[6];

          if(subpedvalue_)
          {
             fBin5 -= calibrations.pedestal((*ihbhe)[4].capid());

             fBin67 -= (calibrations.pedestal((*ihbhe)[5].capid())
                      + calibrations.pedestal((*ihbhe)[6].capid()));
          }
          else 
          {
            fBin5 -= pedvalue;
            fBin67 -= 2*pedvalue;
          }

          //fBin12 is a pedestal, others are percentages
          if(fDigiSum > 0)
          {
            fBin5 /= fDigiSum;
            fBin67 /= fDigiSum;
          }

          monitor()->fillBin5Frac(fBin5);
          monitor()->fillBin67Frac(fBin67);
        }
	      ndigis++;
	    }
    }

        
    
      edm::Handle<PCaloHitContainer> hcalHits ;
     // iEvent.getByLabel("SimG4Object","HcalHits",hcalHits);
      iEvent.getByLabel("g4SimHits","HcalHits",hcalHits);
      
      const PCaloHitContainer * simhitResult = hcalHits.product () ;
      
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

  monitor()->fillDigiSimhit(fEnergySimHits, fAdcSum);
  monitor()->fillRatioDigiSimhit(fAdcSum/fEnergySimHits);
  monitor()->fillDigiSimhitProfile(fEnergySimHits, fAdcSum);
  monitor()->fillSumDigis(fAdcSum);
  monitor()->fillSumDigis_noise(fAdcSum);
  monitor()->fillNDigis(ndigis);
}



HcalDigiTester::HcalDigiTester(const edm::ParameterSet& iConfig)
: dbe_(edm::Service<DaqMonitorBEInterface>().operator->()),
  outputFile_(iConfig.getUntrackedParameter<string>("outputFile", "")),
  hcalselector_(iConfig.getUntrackedParameter<string>("hcalselector", "all")),
  subpedvalue_(iConfig.getUntrackedParameter<bool>("subpedvalue", "true")),
  monitors_()
{
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal Digi Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    LogInfo("OutputInfo") << " Hcal Digi Task histograms will NOT be saved";
  }


 if ( dbe_ ) {
   dbe_->setCurrentFolder("HcalDigiTask");
 }
}
   



HcalDigiTester::~HcalDigiTester()
{
  cout << " outputFile_.size() =  " << outputFile_.size() << endl;
  cout << " dbe_ = " << dbe_ << endl; 
 if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}

void HcalDigiTester::endJob() {
 cout << " outputFile_.size() =  " << outputFile_.size() << endl;
  cout << " dbe_ = " << dbe_ << endl; 
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
  //  reco<HBHEDataFrame>(iEvent,iSetup);
  pedvalue = 4.5;
  if (hcalselector_ == "HB" ) reco<HBHEDataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HE" ) reco<HBHEDataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HO" ) reco<HODataFrame>(iEvent,iSetup);
  pedvalue = 1.73077;
  if (hcalselector_ == "HF" ) reco<HFDataFrame>(iEvent,iSetup);  
                                                          
  if (hcalselector_ == "noise") 
    {
      pedvalue = 4.5;
      hcalselector_ = "HB";
      reco<HBHEDataFrame>(iEvent,iSetup);
      hcalselector_ = "HE";
      reco<HBHEDataFrame>(iEvent,iSetup);
      hcalselector_ = "HO";
      reco<HODataFrame>(iEvent,iSetup);
      hcalselector_ = "HF";
      pedvalue = 1.73077;
      reco<HFDataFrame>(iEvent,iSetup);
      hcalselector_ = "noise";
    }

}

