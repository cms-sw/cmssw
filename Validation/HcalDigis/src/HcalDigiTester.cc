#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Validation/HcalDigis/interface/HcalDigiTester.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

#include "DQMServices/Core/interface/DQMStore.h"

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

#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

template<class Digi>

void HcalDigiTester::reco(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  double fphi_mc = -999999.;  // phi of initial particle from HepMC
  double feta_mc = -999999.;  // eta of initial particle from HepMC
  
  bool MC = false;
  
  // double deltaR = 0.05;
  
  edm::Handle<edm::HepMCProduct> evtMC;
  //  iEvent.getByLabel("VtxSmeared",evtMC);
  iEvent.getByLabel("source",evtMC);
  if (!evtMC.isValid()) {
    MC=false;
    std::cout << "no HepMCProduct found" << std::endl;    
  } else {
    MC=true;
    //    std::cout << "source HepMCProduct found"<< std::endl;
  }
  
  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evtMC->GetEvent()));
  for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	p != myGenEvent->particles_end(); ++p ) {
    fphi_mc = (*p)->momentum().phi();
    feta_mc = (*p)->momentum().eta();
  }


  typename   edm::Handle<edm::SortedCollection<Digi> > digiCollection;
  typename edm::SortedCollection<Digi>::const_iterator digiItr;
  
   
  // ADC2fC 

  const HcalQIEShape* shape = conditions->getHcalShape();
  HcalCalibrations calibrations;
  CaloSamples tool;


  iEvent.getByLabel (inputTag_, digiCollection);

  int subdet = 0;
  if (hcalselector_ == "HB"  ) subdet = 1;
  if (hcalselector_ == "HE"  ) subdet = 2;
  if (hcalselector_ == "HO"  ) subdet = 3;
  if (hcalselector_ == "HF"  ) subdet = 4; 

  if(subdet == 1) nevent1++;
  if(subdet == 2) nevent2++;
  if(subdet == 3) nevent3++;
  if(subdet == 4) nevent4++;
  
  int zsign = 0;
  if (zside_ == "+")  zsign =  1;
  if (zside_ == "-")  zsign = -1;

  int ndigis = 0;
  //  amplitude for signal cell at diff. depths
  double ampl1    = 0.;
  double ampl2    = 0.;
  double ampl3    = 0.;
  double ampl4    = 0.;
  double ampl_all_depths = 0.;

  /*
  std::cout << " HcalDigiTester::reco :  "
	    << "subdet=" << subdet << "  noise="<< noise_ << std::endl;
  */

  // CYCLE OVER CELLS ========================================================

  for (digiItr=digiCollection->begin();digiItr!=digiCollection->end();digiItr++) {
    
    HcalDetId cell(digiItr->id()); 
    int depth = cell.depth();
    int iphi  = cell.iphi()-1;
    int ieta  = cell.ieta();
    if(ieta > 0) ieta--;
    
    // Gains, pedestals (once !) and only for "noise" case  
    if ( ((nevent1 == 1 && subdet == 1) || 
	  (nevent2 == 1 && subdet == 2) ||
	  (nevent3 == 1 && subdet == 3) ||
	  (nevent4 == 1 && subdet == 4)) && noise_ == 1) { 

      HcalGenericDetId hcalGenDetId(digiItr->id());
      const HcalPedestal* pedestal = conditions->getPedestal(hcalGenDetId);
      const HcalGain*  gain = conditions->getGain(hcalGenDetId); 
      const HcalGainWidth* gainWidth = 
	conditions->getGainWidth(hcalGenDetId); 
      const HcalPedestalWidth* pedWidth =
	conditions-> getPedestalWidth(hcalGenDetId);  
      
      double gainValue = gain->getValue(0);
      double gainWidthValue = gainWidth->getValue(0);
      
      /*     
	std::cout << " ieta, iphi, depth : " 
	<< ieta << " " << iphi << " " << depth 
	<< "  gain " << gainValue << "  gainWidth " << gainWidthValue
	<< std::endl;
      */
      
      double pedValue0 = pedestal->getValue(0);
      double pedValue1 = pedestal->getValue(1);
      double pedValue2 = pedestal->getValue(2);
      double pedValue3 = pedestal->getValue(3);
      
      monitor()->fillmePedCapId0(pedValue0);
      monitor()->fillmePedCapId1(pedValue1);
      monitor()->fillmePedCapId2(pedValue2);
      monitor()->fillmePedCapId3(pedValue3);
      
      double pedWidth0 = pedWidth->getWidth(0);
      double pedWidth1 = pedWidth->getWidth(1);
      double pedWidth2 = pedWidth->getWidth(2);
      double pedWidth3 = pedWidth->getWidth(3);
      
      monitor()->fillmePedWidthCapId0(pedWidth0);
      monitor()->fillmePedWidthCapId1(pedWidth1);
      monitor()->fillmePedWidthCapId2(pedWidth2);
      monitor()->fillmePedWidthCapId3(pedWidth3);
      
      if (depth == 1) {
	monitor()->fillmeGainDepth1     (gainValue);
	monitor()->fillmeGainWidthDepth1(gainWidthValue);
	monitor()->fillmeGainMap1 (double(ieta), double(iphi), gainValue);
	monitor()->fillmePwidthMap1(double(ieta), double(iphi), pedWidth0) ;  
      }
      if (depth == 2) {
	monitor()->fillmeGainDepth2     (gainValue);
	monitor()->fillmeGainWidthDepth2(gainWidthValue);
	monitor()->fillmeGainMap2 (double(ieta), double(iphi), gainValue);
	monitor()->fillmePwidthMap2(double(ieta), double(iphi), pedWidth0) ;  
      }
      if (depth == 3) {
	monitor()->fillmeGainDepth3     (gainValue);
	monitor()->fillmeGainWidthDepth3(gainWidthValue);
	monitor()->fillmeGainMap3 (double(ieta), double(iphi), gainValue);
	monitor()->fillmePwidthMap3(double(ieta), double(iphi), pedWidth0) ;  
      }
      if (depth == 4) {
	monitor()->fillmeGainDepth4     (gainValue);
	monitor()->fillmeGainWidthDepth4(gainWidthValue);
	monitor()->fillmeGainMap4 (double(ieta), double(iphi), gainValue);
	monitor()->fillmePwidthMap4(double(ieta), double(iphi), pedWidth0) ;  
      }
    }     // end of event #1 
    
    
    // No-noise case, only subdet selected  =================================
    if ( cell.subdet() == subdet && noise_ == 0 ) {   
      
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
      double fEta = cellGeometry->getPosition ().eta () ;
      double fPhi = cellGeometry->getPosition ().phi () ;
      int depth   = cell.depth();
      
      // old version 
      //      conditions->makeHcalCalibration(cell, &calibrations);
      HcalCalibrations calibrations = conditions->getHcalCalibrations(cell);

      const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
      HcalCoderDb coder (*channelCoder, *shape);
      coder.adc2fC(*digiItr,tool);
      
      double noiseADC =  (*digiItr)[0].adc();     
      double noisefC  =  tool[0];     
      
      if(depth == 1) { 
	monitor()->fillmeADC0_depth1  (noiseADC);
	monitor()->fillmeADC0fC_depth1(noisefC);
      }
      if(depth == 2) { 
	monitor()->fillmeADC0_depth2  (noiseADC);
	monitor()->fillmeADC0fC_depth2(noisefC);
      }
      if(depth == 3) { 
	monitor()->fillmeADC0_depth3  (noiseADC);
	monitor()->fillmeADC0fC_depth3(noisefC);
      }
      if(depth == 4) { 
	monitor()->fillmeADC0_depth4  (noiseADC);
	monitor()->fillmeADC0fC_depth4(noisefC);
      }
      
      // Cycle on time slices 
      
      double ampl = 0.;
      int closen  = 0;
      if(fabs(feta_mc-fEta) < 0.087/2. && acos(cos(fphi_mc-fPhi))<0.087/2.)
	closen = 1;

      for  (int ii=0;ii<tool.size();ii++) {
	int capid  = (*digiItr)[ii].capid();
	double val = (tool[ii]-calibrations.pedestal(capid));
	
	monitor()->fillmeAll10slices(double(ii), val);

 	
	if( closen == 1 && ( ieta * zsign >= 0 )) { 
	  monitor()->fillmeSignalTimeSlice(double(ii), val);
	}

	// HB/HE/HO
	if (subdet != 4 && ii>=4 && ii<=7) { 
	  ampl += val;	  
	  if( closen == 1 && ( ieta * zsign >= 0 )) { 
	    
	    if(depth == 1) ampl1 += val;   
	    if(depth == 2) ampl2 += val;   
	    if(depth == 3) ampl3 += val;   
	    if(depth == 4) ampl4 += val;
	    
	  }
	}
	
	// HF
	if (subdet == 4 && ii==3 )	{
	  ampl += val;
	  if( closen == 1 && ( ieta * zsign >= 0 )) { 
	    //	  if( dR(feta_mc, fphi_mc, fEta, fPhi) < deltaR) {  
	    
	    if(depth == 1)  ampl1 += val;
	    if(depth == 2)  ampl2 += val;
	    if(depth == 3)  ampl3 += val;
	    if(depth == 4)  ampl4 += val;

	  }
	}
      }
      // end of time bucket sample      
      
      monitor()->fillmeAmplIetaIphi(double(ieta),double(iphi), ampl);
      monitor()->fillmeSumAmp      (ampl);
      
      
      if(ampl > 10.) ndigis++;
      
      // fraction 5,6 bins if ampl. is big.
      if(ampl1 > 50. &&  depth == 1 && closen == 1 ) { 
	  double fBin5  = tool[4] - calibrations.pedestal((*digiItr)[4].capid());
	double fBin67 = tool[5] + tool[6] 
	  - calibrations.pedestal((*digiItr)[5].capid())
	  - calibrations.pedestal((*digiItr)[6].capid());
	fBin5  /= ampl1;
	fBin67 /= ampl1;
	monitor()->fillmeBin5Frac (fBin5);
	monitor()->fillmeBin67Frac(fBin67);
      }
      
    }   
  }    // End of CYCLE OVER CELLS =============================================
  
  
  if ( subdet != 0 && noise_ == 0) { // signal only 

    ampl_all_depths = ampl1+ampl2+ampl3+ampl4;
    monitor()->fillmeSignalAmp (ampl_all_depths); 
    monitor()->fillmeSignalAmp1(ampl1); 
    monitor()->fillmeSignalAmp2(ampl2); 
    monitor()->fillmeSignalAmp3(ampl3); 
    monitor()->fillmeSignalAmp4(ampl4); 
    
    monitor()->fillmenDigis(ndigis);
    
    // SimHits 
    
    edm::Handle<edm::PCaloHitContainer> hcalHits ;
    iEvent.getByLabel("g4SimHits","HcalHits",hcalHits);
    
    const edm::PCaloHitContainer * simhitResult = hcalHits.product () ;
    
    double ehits  = 0.; 
    double ehits1 = 0.; 
    double ehits2 = 0.; 
    double ehits3 = 0.; 
    double ehits4 = 0.; 
 
    for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin ();         simhits != simhitResult->end () ;  ++simhits) {
      
      HcalDetId cell(simhits->id());
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
      double fEta  = cellGeometry->getPosition ().eta () ;
      double fPhi  = cellGeometry->getPosition ().phi () ;
      
      if (cell.subdet() == subdet &&  
	  (fabs(feta_mc-fEta) < 0.087/2. && acos(cos(fphi_mc-fPhi))<0.087/2.)){  
	int depth = cell.depth();
	double en = simhits->energy();
	
	ehits += en;
	if(depth == 1)  ehits1 += en; 
	if(depth == 2)  ehits2 += en; 
	if(depth == 3)  ehits3 += en; 
	if(depth == 4)  ehits4 += en; 
      }
    }
    
    
    monitor()->fillmeDigiSimhit (ehits,  ampl_all_depths);
    monitor()->fillmeDigiSimhit1(ehits1, ampl1);
    monitor()->fillmeDigiSimhit2(ehits2, ampl2);
    monitor()->fillmeDigiSimhit3(ehits3, ampl3);
    monitor()->fillmeDigiSimhit4(ehits4, ampl4);
    
    monitor()->fillmeDigiSimhitProfile (ehits,  ampl_all_depths);
    monitor()->fillmeDigiSimhitProfile1(ehits1, ampl1);
    monitor()->fillmeDigiSimhitProfile2(ehits2, ampl2);
    monitor()->fillmeDigiSimhitProfile3(ehits3, ampl3);
    monitor()->fillmeDigiSimhitProfile4(ehits4, ampl4);
    
    if(ehits  > 0) monitor()->fillmeRatioDigiSimhit (ampl_all_depths / ehits);
    if(ehits1 > 0) monitor()->fillmeRatioDigiSimhit1(ampl1 / ehits1);
    if(ehits2 > 0) monitor()->fillmeRatioDigiSimhit2(ampl2 / ehits2);
    if(ehits3 > 0) monitor()->fillmeRatioDigiSimhit3(ampl3 / ehits3);
    if(ehits4 > 0) monitor()->fillmeRatioDigiSimhit4(ampl4 / ehits4);
    
  }  
}


HcalDigiTester::HcalDigiTester(const edm::ParameterSet& iConfig)
  : dbe_(edm::Service<DQMStore>().operator->()),
    inputTag_(iConfig.getParameter<edm::InputTag>("digiLabel")),
    outputFile_(iConfig.getUntrackedParameter<std::string>("outputFile", "")),
    hcalselector_(iConfig.getUntrackedParameter<std::string>("hcalselector", "all")),
    zside_(iConfig.getUntrackedParameter<std::string>("zside", "*")),
    monitors_()
{
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal Digi Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal Digi Task histograms will NOT be saved";
  }


}
   

HcalDigiTester::~HcalDigiTester() { 
  std::cout << " outputFile_.size() =  " << outputFile_.size() << std::endl;
  std::cout << " dbe_ = " << dbe_ << std::endl; 
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void HcalDigiTester::endJob() { }

void HcalDigiTester::beginJob(const edm::EventSetup& c){

  nevent1 = 0;
  nevent2 = 0;
  nevent3 = 0;
  nevent4 = 0;

}


HcalSubdetDigiMonitor * HcalDigiTester::monitor()
{
  std::map<std::string, HcalSubdetDigiMonitor*>::iterator monitorItr
    = monitors_.find(hcalselector_);

  if(monitorItr == monitors_.end())
    {
      HcalSubdetDigiMonitor* m = new HcalSubdetDigiMonitor(dbe_, hcalselector_, noise_);
      std::pair<std::string, HcalSubdetDigiMonitor*> mapElement(
								hcalselector_, m);
      monitorItr = monitors_.insert(mapElement).first;
    }
  return monitorItr->second;
}

void 
HcalDigiTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  iSetup.get<CaloGeometryRecord>().get (geometry);
  iSetup.get<HcalDbRecord>().get(conditions);

  if (hcalselector_ == "HB" ) reco<HBHEDataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HE" ) reco<HBHEDataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HO" ) reco<HODataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HF" ) reco<HFDataFrame>(iEvent,iSetup);  

  noise_ = 0;                                                          
  if (hcalselector_ == "noise") 
    {
      noise_ = 1;
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
double HcalDigiTester::dR(double eta1, double phi1, double eta2, double phi2) { 
  double PI = 3.1415926535898;
  double deltaphi= phi1 - phi2;
  if( phi2 > phi1 ) { deltaphi= phi2 - phi1;}
  if(deltaphi > PI) { deltaphi = 2.*PI - deltaphi;}
  double deltaeta = eta2 - eta1;
  double tmp = sqrt(deltaeta* deltaeta + deltaphi*deltaphi);
  return tmp;
}

