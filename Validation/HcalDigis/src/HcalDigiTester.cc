#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

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
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
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
  iEvent.getByLabel("generator",evtMC);
  if (!evtMC.isValid()) {
    MC=false;
    std::cout << "no HepMCProduct found" << std::endl;    
  } else {
    MC=true;
    //    std::cout << "source HepMCProduct found"<< std::endl;
  }
  
  // MC particle with highest pt is taken as a direction reference  
  double maxPt = -99999.;
  int npart    = 0;
  const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	p != myGenEvent->particles_end(); ++p ) {
    double phip = (*p)->momentum().phi();
    double etap = (*p)->momentum().eta();
    //    phi_MC = phip;
    //    eta_MC = etap;
    double pt  = (*p)->momentum().perp();
    if(pt > maxPt) { npart++; maxPt = pt; fphi_mc = phip; feta_mc = etap; }
  }
  //  std::cout << "*** Max pT = " << maxPt <<  std::endl;  

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

 
  //  std::cout << " HcalDigiTester::reco :  "
  //	    << "subdet=" << subdet << "  noise="<< noise_ << std::endl;
 

  // CYCLE OVER CELLS ========================================================

  for (digiItr=digiCollection->begin();digiItr!=digiCollection->end();digiItr++) {
    
    HcalDetId cell(digiItr->id()); 
    int depth = cell.depth();
    int iphi  = cell.iphi()-1;
    int ieta  = cell.ieta();
    int sub   = cell.subdet();
    if(ieta > 0) ieta--;
    
    // Gains, pedestals (once !) and only for "noise" case  
    if ( ((nevent1 == 1 && subdet == 1) || 
	  (nevent2 == 1 && subdet == 2) ||
	  (nevent3 == 1 && subdet == 3) ||
	  (nevent4 == 1 && subdet == 4)) && noise_ == 1 && sub == subdet) { 

      HcalGenericDetId hcalGenDetId(digiItr->id());
      const HcalPedestal* pedestal = conditions->getPedestal(hcalGenDetId);
      const HcalGain*  gain = conditions->getGain(hcalGenDetId); 
      const HcalGainWidth* gainWidth = 
	conditions->getGainWidth(hcalGenDetId); 
      const HcalPedestalWidth* pedWidth =
	conditions-> getPedestalWidth(hcalGenDetId);  
      
      double gainValue0 = gain->getValue(0);
      double gainValue1 = gain->getValue(1);
      double gainValue2 = gain->getValue(2);
      double gainValue3 = gain->getValue(3);

      double gainWidthValue0 = gainWidth->getValue(0);
      double gainWidthValue1 = gainWidth->getValue(1);
      double gainWidthValue2 = gainWidth->getValue(2);
      double gainWidthValue3 = gainWidth->getValue(3);
      
      /*     
	std::cout << " ieta, iphi, depth : " 
	<< ieta << " " << iphi << " " << depth 
	<< "  gain0 " << gainValue0 << "  gainWidth0 " << gainWidthValue0
	<< std::endl;
      */
      
      double pedValue0 = pedestal->getValue(0);
      double pedValue1 = pedestal->getValue(1);
      double pedValue2 = pedestal->getValue(2);
      double pedValue3 = pedestal->getValue(3);
      
      double pedWidth0 = pedWidth->getWidth(0);
      double pedWidth1 = pedWidth->getWidth(1);
      double pedWidth2 = pedWidth->getWidth(2);
      double pedWidth3 = pedWidth->getWidth(3);
      
      if (depth == 1) {
	monitor()->fillmeGain0Depth1(gainValue0);
	monitor()->fillmeGain1Depth1(gainValue1);
	monitor()->fillmeGain2Depth1(gainValue2);
	monitor()->fillmeGain3Depth1(gainValue3);

	monitor()->fillmeGainWidth0Depth1(gainWidthValue0);
	monitor()->fillmeGainWidth1Depth1(gainWidthValue1);
	monitor()->fillmeGainWidth2Depth1(gainWidthValue2);
	monitor()->fillmeGainWidth3Depth1(gainWidthValue3);

	monitor()->fillmePed0Depth1(pedValue0);
	monitor()->fillmePed1Depth1(pedValue1);
	monitor()->fillmePed2Depth1(pedValue2);
	monitor()->fillmePed3Depth1(pedValue3);

        monitor()->fillmePedWidth0Depth1(pedWidth0);
        monitor()->fillmePedWidth1Depth1(pedWidth1);
        monitor()->fillmePedWidth2Depth1(pedWidth2);
        monitor()->fillmePedWidth3Depth1(pedWidth3);

	monitor()->fillmeGainMap1  (double(ieta), double(iphi), gainValue0);
	monitor()->fillmePwidthMap1(double(ieta), double(iphi), pedWidth0) ;  
      }

      if (depth == 2) {
	monitor()->fillmeGain0Depth2(gainValue0);
	monitor()->fillmeGain1Depth2(gainValue1);
	monitor()->fillmeGain2Depth2(gainValue2);
	monitor()->fillmeGain3Depth2(gainValue3);

	monitor()->fillmeGainWidth0Depth2(gainWidthValue0);
	monitor()->fillmeGainWidth1Depth2(gainWidthValue1);
	monitor()->fillmeGainWidth2Depth2(gainWidthValue2);
	monitor()->fillmeGainWidth3Depth2(gainWidthValue3);

	monitor()->fillmePed0Depth2(pedValue0);
	monitor()->fillmePed1Depth2(pedValue1);
	monitor()->fillmePed2Depth2(pedValue2);
	monitor()->fillmePed3Depth2(pedValue3);

        monitor()->fillmePedWidth0Depth2(pedWidth0);
        monitor()->fillmePedWidth1Depth2(pedWidth1);
        monitor()->fillmePedWidth2Depth2(pedWidth2);
        monitor()->fillmePedWidth3Depth2(pedWidth3);

	monitor()->fillmeGainMap1  (double(ieta), double(iphi), gainValue0);
	monitor()->fillmePwidthMap1(double(ieta), double(iphi), pedWidth0) ;  
      }

      if (depth == 3) {
	monitor()->fillmeGain0Depth3(gainValue0);
	monitor()->fillmeGain1Depth3(gainValue1);
	monitor()->fillmeGain2Depth3(gainValue2);
	monitor()->fillmeGain3Depth3(gainValue3);

	monitor()->fillmeGainWidth0Depth3(gainWidthValue0);
	monitor()->fillmeGainWidth1Depth3(gainWidthValue1);
	monitor()->fillmeGainWidth2Depth3(gainWidthValue2);
	monitor()->fillmeGainWidth3Depth3(gainWidthValue3);

	monitor()->fillmePed0Depth3(pedValue0);
	monitor()->fillmePed1Depth3(pedValue1);
	monitor()->fillmePed2Depth3(pedValue2);
	monitor()->fillmePed3Depth3(pedValue3);

        monitor()->fillmePedWidth0Depth3(pedWidth0);
        monitor()->fillmePedWidth1Depth3(pedWidth1);
        monitor()->fillmePedWidth2Depth3(pedWidth2);
        monitor()->fillmePedWidth3Depth3(pedWidth3);

	monitor()->fillmeGainMap1  (double(ieta), double(iphi), gainValue0);
	monitor()->fillmePwidthMap1(double(ieta), double(iphi), pedWidth0) ;  
      }

      if (depth == 4) {
	monitor()->fillmeGain0Depth4(gainValue0);
	monitor()->fillmeGain1Depth4(gainValue1);
	monitor()->fillmeGain2Depth4(gainValue2);
	monitor()->fillmeGain3Depth4(gainValue3);

	monitor()->fillmeGainWidth0Depth4(gainWidthValue0);
	monitor()->fillmeGainWidth1Depth4(gainWidthValue1);
	monitor()->fillmeGainWidth2Depth4(gainWidthValue2);
	monitor()->fillmeGainWidth3Depth4(gainWidthValue3);

	monitor()->fillmePed0Depth4(pedValue0);
	monitor()->fillmePed1Depth4(pedValue1);
	monitor()->fillmePed2Depth4(pedValue2);
	monitor()->fillmePed3Depth4(pedValue3);

        monitor()->fillmePedWidth0Depth4(pedWidth0);
        monitor()->fillmePedWidth1Depth4(pedWidth1);
        monitor()->fillmePedWidth2Depth4(pedWidth2);
        monitor()->fillmePedWidth3Depth4(pedWidth3);

	monitor()->fillmeGainMap1  (double(ieta), double(iphi), gainValue0);
	monitor()->fillmePwidthMap1(double(ieta), double(iphi), pedWidth0) ;  

      }
    }     // end of event #1 
    
    
    // No-noise case, only subdet selected  =================================
    if ( sub == subdet && noise_ == 0 ) {   
      
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
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void HcalDigiTester::endJob() { }

void HcalDigiTester::beginJob() {

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

  noise_ = 0;
                                                          
  if (hcalselector_ == "HB" ) reco<HBHEDataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HE" ) reco<HBHEDataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HO" ) reco<HODataFrame>(iEvent,iSetup);
  if (hcalselector_ == "HF" ) reco<HFDataFrame>(iEvent,iSetup);  

  if (hcalselector_ == "noise") {
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

// New block !!!!
  if (hcalselector_ == "all") {
   noise_ = 0;

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


DEFINE_SEAL_MODULE ();
DEFINE_ANOTHER_FWK_MODULE (HcalDigiTester) ;
