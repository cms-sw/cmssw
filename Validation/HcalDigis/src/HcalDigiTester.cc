#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
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
  
  
  typename   edm::Handle<edm::SortedCollection<Digi> > digiCollection;
  typename edm::SortedCollection<Digi>::const_iterator digiItr;
     
  // ADC2fC 
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
  double ampl1_c    = 0.;
  double ampl2_c    = 0.;
  double ampl3_c    = 0.;
  double ampl4_c    = 0.;
  double ampl_c     = 0.;

  // is set to 1 if "seed" SimHit is found 
  int seedSimHit  = 0;
 
  //  std::cout << " HcalDigiTester::reco :  "
  // 	    << "subdet=" << subdet << "  noise="<< noise_ << std::endl;
 
  int ieta_Sim    =  9999;
  int iphi_Sim    =  9999;
  double emax_Sim = -9999.;

    
  // SimHits MC only
  if( mc_ == "yes") {
    edm::Handle<edm::PCaloHitContainer> hcalHits ;
    iEvent.getByLabel("g4SimHits","HcalHits",hcalHits); 
    const edm::PCaloHitContainer * simhitResult = hcalHits.product () ;
    
    
    if ( subdet != 0 && noise_ == 0) { // signal only SimHits
      
      for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin ();         simhits != simhitResult->end () ;  ++simhits) {
	
	HcalDetId cell(simhits->id());
	double en    = simhits->energy();
	int sub      = cell.subdet();
	int ieta     = cell.ieta();
	if(ieta > 0) ieta--;
	int iphi     = cell.iphi()-1; 
	
	
	if(en > emax_Sim && sub == subdet) {
	  emax_Sim = en;
	  ieta_Sim = ieta;
	  iphi_Sim = iphi;            
	  // to limit "seed" SimHit energy in case of "multi" event  
	  if (mode_ == "multi" && 
           ((sub == 4 && en < 100. && en > 1.) 
	    || ((sub !=4) && en < 1. && en > 0.02))) 
	    {
	      seedSimHit = 1;            
	      break;   
	    }
	}
	
      } // end of SimHits cycle
      
      
      // found highest-energy SimHit for single-particle 
      if(mode_ != "multi" && emax_Sim > 0.) seedSimHit = 1;
    }   // end of SimHits
    
  } // end of mc_ == "yes"

  // CYCLE OVER CELLS ========================================================
  int Ndig = 0;

  /*
  std::cout << " HcalDigiTester::reco :     nevent 1,2,3,4 = "
	    << nevent1 << " " << nevent2 << " " << nevent3 << " " 
	    << nevent4 << std::endl;
  */

  for (digiItr=digiCollection->begin();digiItr!=digiCollection->end();digiItr++) {
    
    HcalDetId cell(digiItr->id()); 
    int depth = cell.depth();
    int iphi  = cell.iphi()-1;
    int ieta  = cell.ieta();
    if(ieta > 0) ieta--;
    int sub   = cell.subdet();


  //  amplitude for signal cell at diff. depths
    double ampl     = 0.;
    double ampl1    = 0.;
    double ampl2    = 0.;
    double ampl3    = 0.;
    double ampl4    = 0.;
    
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
      


      // some printout
      /*
      std::cout <<  " subdet = " << sub << "  ieta, iphi, depth : " 
		<< ieta << " " << iphi << " " << depth 
		<< "  gain0 " << gainValue0 << "  gainWidth0 " 
		<< gainWidthValue0
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

	//        std::cout <<  "          depth = " << depth << std::endl;
    
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

	//        std::cout <<  "          depth = " << depth << std::endl;

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

	monitor()->fillmeGainMap2  (double(ieta), double(iphi), gainValue0);
	monitor()->fillmePwidthMap2(double(ieta), double(iphi), pedWidth0) ;  
      }

      if (depth == 3) {

	//        std::cout <<  "          depth = " << depth << std::endl;

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

	monitor()->fillmeGainMap3  (double(ieta), double(iphi), gainValue0);
	monitor()->fillmePwidthMap3(double(ieta), double(iphi), pedWidth0) ;  
      }

      if (depth == 4) {

	//        std::cout <<  "          depth = " << depth << std::endl;

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

	monitor()->fillmeGainMap4  (double(ieta), double(iphi), gainValue0);
	monitor()->fillmePwidthMap4(double(ieta), double(iphi), pedWidth0) ;  

      }

    }     // end of event #1 
    //std::cout << "==== End of event noise block in cell cycle"  << std::endl;



    if ( sub == subdet)  Ndig++;  // subdet number of digi
    
// No-noise case, only single  subdet selected  ===========================

    if ( sub == subdet && noise_ == 0 ) {   

      
      HcalCalibrations calibrations = conditions->getHcalCalibrations(cell);

      const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
      const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
      HcalCoderDb coder (*channelCoder, *shape);
      coder.adc2fC(*digiItr,tool);
      
      double noiseADC =  (*digiItr)[0].adc();     
      double noisefC  =  tool[0];     
      
      // noise evaluations from "pre-samples"
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

      // OCCUPANCY maps filling
      double deta = double(ieta);
      double dphi = double(iphi);
      if(depth == 1)
      monitor()->fillmeOccupancy_map_depth1(deta, dphi); 
      if(depth == 2)
      monitor()->fillmeOccupancy_map_depth2(deta, dphi); 
      if(depth == 3)
      monitor()->fillmeOccupancy_map_depth3(deta, dphi); 
      if(depth == 4)
      monitor()->fillmeOccupancy_map_depth4(deta, dphi); 
      
      // Cycle on time slices
      // - for each Digi 
      // - for one Digi with max SimHits E in subdet
      
      int closen = 0;   // =1 if 1) seedSimHit = 1 and 2) the cell is the same
      if(ieta == ieta_Sim && iphi == iphi_Sim ) closen = seedSimHit;

      for  (int ii=0;ii<tool.size();ii++) {
	int capid  = (*digiItr)[ii].capid();
        // single ts amplitude
	double val = (tool[ii]-calibrations.pedestal(capid));

	if (val > 10.) {
	  if (depth == 1) 
	    monitor()->fillmeAll10slices_depth1(double(ii), val);
          else 
	    monitor()->fillmeAll10slices_depth2(double(ii), val);
	}
	if (val > 100.) {
	  if (depth == 1) 
	    monitor()->fillmeAll10slices1D_depth1(double(ii), val);
          else 
	    monitor()->fillmeAll10slices1D_depth2(double(ii), val);
	}
 	
	if( closen == 1 &&( ieta * zsign >= 0 )) { 
	  monitor()->fillmeSignalTimeSlice(double(ii), val);
	}


	// HB/HE/HO
	if (subdet != 4 && ii>=4 && ii<=7) { 
	  ampl += val;	  
	  if(depth == 1) ampl1 += val;   
	  if(depth == 2) ampl2 += val;   
	  if(depth == 3) ampl3 += val;   
	  if(depth == 4) ampl4 += val;
	  
	  if( closen == 1 && ( ieta * zsign >= 0 )) { 
            ampl_c += val;	  
	    if(depth == 1) ampl1_c += val;   
	    if(depth == 2) ampl2_c += val;   
	    if(depth == 3) ampl3_c += val;   
	    if(depth == 4) ampl4_c += val;
	    
	  }
	}
	
	// HF
	if (subdet == 4 && ii==3 )	{
	  ampl += val;	    
	  if(depth == 1)  ampl1 += val;
	  if(depth == 2)  ampl2 += val;
	  if(depth == 3)  ampl3 += val;
	  if(depth == 4)  ampl4 += val;
	  if( closen == 1 && ( ieta * zsign >= 0 )) { 	    
	    ampl_c += val;	    
	    if(depth == 1)  ampl1_c += val;
	    if(depth == 2)  ampl2_c += val;
	    if(depth == 3)  ampl3_c += val;
	    if(depth == 4)  ampl4_c += val;

	  }
	}
      }
      // end of time bucket sample      
      
      monitor()->fillmeAmplIetaIphi1(double(ieta),double(iphi), ampl1);
      monitor()->fillmeAmplIetaIphi2(double(ieta),double(iphi), ampl2);
      monitor()->fillmeAmplIetaIphi3(double(ieta),double(iphi), ampl3);
      monitor()->fillmeAmplIetaIphi4(double(ieta),double(iphi), ampl4);
      monitor()->fillmeSumAmp (ampl);
      
      
      if(ampl1 > 10. || ampl2 > 10.  || ampl3 > 10.  || ampl4 > 10. ) ndigis++;
      
      // fraction 5,6 bins if ampl. is big.
      if(ampl1 > 30. &&  depth == 1 && closen == 1 ) { 
	  double fBin5  = tool[4] - calibrations.pedestal((*digiItr)[4].capid());
	double fBin67 = tool[5] + tool[6] 
	  - calibrations.pedestal((*digiItr)[5].capid())
	  - calibrations.pedestal((*digiItr)[6].capid());
	fBin5  /= ampl1;
	fBin67 /= ampl1;
	monitor()->fillmeBin5Frac (fBin5);
	monitor()->fillmeBin67Frac(fBin67);
      }

      monitor()->fillmeSignalAmp (ampl); 
      monitor()->fillmeSignalAmp1(ampl1); 
      monitor()->fillmeSignalAmp2(ampl2); 
      monitor()->fillmeSignalAmp3(ampl3); 
      monitor()->fillmeSignalAmp4(ampl4); 
    
      
    }   
  }    // End of CYCLE OVER CELLS =============================================
  
  
  if ( subdet != 0 && noise_ == 0) { // signal only, once per event 

    monitor()->fillmenDigis(ndigis);
    
    // SimHits once again !!!
    double eps    = 1.e-3;
    double ehits  = 0.; 
    double ehits1 = 0.; 
    double ehits2 = 0.; 
    double ehits3 = 0.; 
    double ehits4 = 0.; 
 
    if(mc_ == "yes") {
      edm::Handle<edm::PCaloHitContainer> hcalHits ;
      iEvent.getByLabel("g4SimHits","HcalHits",hcalHits); 
      const edm::PCaloHitContainer * simhitResult = hcalHits.product () ;
      for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin ();         simhits != simhitResult->end () ;  ++simhits) {
	
	HcalDetId cell(simhits->id());
	int ieta   = cell.ieta();
	if(ieta > 0) ieta--;
	int iphi   = cell.iphi()-1; 
	int sub    = cell.subdet();
	
	// take cell already found to be max energy in a particular subdet
	if (sub == subdet && ieta == ieta_Sim && iphi == iphi_Sim){  
	  int depth = cell.depth();
	  double en = simhits->energy();
	  
	  ehits += en;
	  if(depth == 1)  ehits1 += en; 
	  if(depth == 2)  ehits2 += en; 
	  if(depth == 3)  ehits3 += en; 
	  if(depth == 4)  ehits4 += en; 
	}
      }
      
      if(ehits  > eps) monitor()->fillmeDigiSimhit (ehits,  ampl_c );
      if(ehits1 > eps) monitor()->fillmeDigiSimhit1(ehits1, ampl1_c);
      if(ehits2 > eps) monitor()->fillmeDigiSimhit2(ehits2, ampl2_c);
      if(ehits3 > eps) monitor()->fillmeDigiSimhit3(ehits3, ampl3_c);
      if(ehits4 > eps) monitor()->fillmeDigiSimhit4(ehits4, ampl4_c);
      
      if(ehits  > eps) monitor()->fillmeDigiSimhitProfile (ehits,  ampl_c );
      if(ehits1 > eps) monitor()->fillmeDigiSimhitProfile1(ehits1, ampl1_c);
      if(ehits2 > eps) monitor()->fillmeDigiSimhitProfile2(ehits2, ampl2_c);
      if(ehits3 > eps) monitor()->fillmeDigiSimhitProfile3(ehits3, ampl3_c);
      if(ehits4 > eps) monitor()->fillmeDigiSimhitProfile4(ehits4, ampl4_c);
      
      if(ehits  > eps) monitor()->fillmeRatioDigiSimhit (ampl_c  / ehits);
      if(ehits1 > eps) monitor()->fillmeRatioDigiSimhit1(ampl1_c / ehits1);
      if(ehits2 > eps) monitor()->fillmeRatioDigiSimhit2(ampl2_c / ehits2);
      if(ehits3 > eps) monitor()->fillmeRatioDigiSimhit3(ampl3_c / ehits3);
      if(ehits4 > eps) monitor()->fillmeRatioDigiSimhit4(ampl4_c / ehits4);
    } // end of if(mc_ == "yes")
   
    monitor()->fillmeNdigis(double(Ndig));
    
  } //  end of if( subdet != 0 && noise_ == 0) { // signal only 

}


HcalDigiTester::HcalDigiTester(const edm::ParameterSet& iConfig)
  : dbe_(edm::Service<DQMStore>().operator->()),
    inputTag_(iConfig.getParameter<edm::InputTag>("digiLabel")),
    outputFile_(iConfig.getUntrackedParameter<std::string>("outputFile", "")),
    hcalselector_(iConfig.getUntrackedParameter<std::string>("hcalselector", "all")),
    zside_(iConfig.getUntrackedParameter<std::string>("zside", "*")),
    mode_(iConfig.getUntrackedParameter<std::string>("mode", "multi")),
    mc_(iConfig.getUntrackedParameter<std::string>("mc", "no")),
    monitors_()
{
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal Digi Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal Digi Task histograms will NOT be saved";
  }


}
   

HcalDigiTester::~HcalDigiTester() { }


void HcalDigiTester::endRun() {

 if(noise_ != 1) {

   if( hcalselector_ == "all") {
    hcalselector_ = "HB";
    eval_occupancy();
    hcalselector_ = "HE";
    eval_occupancy();
    hcalselector_ = "HO";
    eval_occupancy();
    hcalselector_ = "HF";
    eval_occupancy();
   }
   else  // one of subsystem only
    eval_occupancy();
 }

}



void HcalDigiTester::endJob() {

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}


  //occupancies evaluation
void HcalDigiTester::eval_occupancy() {    
  
  int nx = 82;
  int ny = 72;
  float cnorm;
  float fev = float (nevtot);
  //    std::cout << "*** nevtot " <<  nevtot << std::endl; 
  
  float sumphi_1, sumphi_2, sumphi_3, sumphi_4;
  float phi_factor;  
  
  for (int i = 1; i <= nx; i++) {
    sumphi_1 = 0.;
    sumphi_2 = 0.;
    sumphi_3 = 0.;
    sumphi_4 = 0.;
    
    for (int j = 1; j <= ny; j++) {
      
      // occupancies
      cnorm = monitor()->getBinContent_depth1(i,j) / fev;   
      monitor()->setBinContent_depth1(i,j,cnorm);
      cnorm = monitor()->getBinContent_depth2(i,j) / fev;   
      monitor()->setBinContent_depth2(i,j,cnorm);
      cnorm = monitor()->getBinContent_depth3(i,j) / fev;   
      monitor()->setBinContent_depth3(i,j,cnorm);
      cnorm = monitor()->getBinContent_depth4(i,j) / fev;   
      monitor()->setBinContent_depth4(i,j,cnorm);

      sumphi_1 += monitor()->getBinContent_depth1(i,j);
      sumphi_2 += monitor()->getBinContent_depth2(i,j);
      sumphi_3 += monitor()->getBinContent_depth3(i,j);
      sumphi_4 += monitor()->getBinContent_depth4(i,j);

    }
    
    int ieta = i - 42;        // -41 -1, 0 40 
    if(ieta >=0 ) ieta +=1;   // -41 -1, 1 41  - to make it detector-like
    
    if(ieta >= -20 && ieta <= 20 )
      {phi_factor = 72.;}
    else {
      if(ieta >= 40 || ieta <= -40 ) {phi_factor = 18.;}
      else 
	phi_factor = 36.;
    }  


    if(ieta >= 0) ieta -= 1; // -41 -1, 0 40  - to bring back to histo num !!!
    double deta =  double(ieta);

    cnorm = sumphi_1 / phi_factor;
    monitor() -> fillmeOccupancy_vs_ieta_depth1(deta, cnorm);      
    cnorm = sumphi_2 / phi_factor;
    monitor() -> fillmeOccupancy_vs_ieta_depth2(deta, cnorm);
    cnorm = sumphi_3 / phi_factor;
    monitor() -> fillmeOccupancy_vs_ieta_depth3(deta, cnorm);
    cnorm = sumphi_4 / phi_factor;
    monitor() -> fillmeOccupancy_vs_ieta_depth4(deta, cnorm);
      
      
  }  // end of i-loop
  
}

void HcalDigiTester::beginJob() {

  nevent1 = 0;
  nevent2 = 0;
  nevent3 = 0;
  nevent4 = 0;

  nevtot  = 0;

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


  //  std::cout << " >>>>> HcalDigiTester::analyze  hcalselector = " 
  //	    << hcalselector_ << std::endl;

  if( hcalselector_ != "all") {
    noise_ = 0;
    
    

    if (hcalselector_ == "HB" ) reco<HBHEDataFrame>(iEvent,iSetup);
    if (hcalselector_ == "HE" ) reco<HBHEDataFrame>(iEvent,iSetup);
    if (hcalselector_ == "HO" ) reco<HODataFrame>(iEvent,iSetup);
    if (hcalselector_ == "HF" ) reco<HFDataFrame>(iEvent,iSetup);  

    if (hcalselector_ == "noise") {
      noise_ = 1;

      //      std::cout << " >>>>> HcalDigiTester::analyze  entering noise " 
      //	    << std::endl;

      
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
    // all subdetectors
  else {
    noise_ = 0;
    
    hcalselector_ = "HB";
    reco<HBHEDataFrame>(iEvent,iSetup);
    hcalselector_ = "HE";
    reco<HBHEDataFrame>(iEvent,iSetup);
    hcalselector_ = "HO";
    reco<HODataFrame>(iEvent,iSetup);
    hcalselector_ = "HF";
    reco<HFDataFrame>(iEvent,iSetup);
    hcalselector_ = "all";    
  }

  nevtot++;

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


DEFINE_FWK_MODULE (HcalDigiTester) ;
