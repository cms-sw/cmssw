#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/HcalDigis/interface/Digi2Raw2Digi.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>


// Qunatities of interest in :
// DataFormats/HcalDigi/test/HcalDigiDump.cc - example of Digi dumping...
// DataFormats/HcalDigi/interface/HcalQIESample.h - adc, capid etc.
// DataFormats/HcalDigi/interface/HBHEDataFrame.h -
// zsMarkAndPass, zsUnsuppressed etc.

template<class Digi>


void Digi2Raw2Digi::compare(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  typename edm::Handle<edm::SortedCollection<Digi> > digiCollection1;
  typename edm::SortedCollection<Digi>::const_iterator digiItr1;
  typename edm::Handle<edm::SortedCollection<Digi> > digiCollection2;
  typename edm::SortedCollection<Digi>::const_iterator digiItr2;

   if(unsuppressed) {  // ZDC
     iEvent.getByLabel ("simHcalUnsuppressedDigis", digiCollection1); 
   }
  else iEvent.getByLabel (inputTag1_, digiCollection1);

  iEvent.getByLabel (inputTag2_, digiCollection2);
 
  int size1 = 0;
  int size2 = 0;

  for (digiItr2=digiCollection2->begin();digiItr2!=digiCollection2->end();digiItr2++) {
    size2++;
 }
  std::cout<<"size2="<<size2<<std::endl;
  // CYCLE over first DIGI collection ======================================

  for (digiItr1=digiCollection1->begin();digiItr1!=digiCollection1->end();digiItr1++) {
    size1++;    

   

    HcalGenericDetId HcalGenDetId(digiItr1->id());
    int tsize =  (*digiItr1).size();
    int match = 0;
 
    if(HcalGenDetId.isHcalZDCDetId()){

      std::cout<<digiItr1->id()<<std::endl;
      
      //for zdc
      HcalZDCDetId element(digiItr1->id());
      int zside =  element.zside();
      int section = element.section();
      int channel = element.channel();
      int gsub = HcalGenDetId.genericSubdet();

      std::cout<< " Zdc genSubdet="<< gsub << " zside=" <<zside
	       << " section= "<< section << " channel " <<channel
	       <<std::endl; 

      for (digiItr2=digiCollection2->begin();digiItr2!=digiCollection2->end();digiItr2++) {
  	HcalZDCDetId element2(digiItr2->id());
	if( element == element2) {
	  match = 1;
	  int identical = 1; 
	  for (int i=0; i<tsize; i++) {
	    double adc  =  (*digiItr1)[i].adc();     
	    int capid   =  (*digiItr1)[i].capid();
	    //	  std::cout << std::endl << "     capid1=" << capid 
	    //                << " adc1=" << adc 
	    //		    << std::endl;
	    double adc2 =  (*digiItr2)[i].adc();     
	    int capid2  =  (*digiItr2)[i].capid();
	    //	  std::cout << "     capid2=" << capid2 << " adc2=" << adc2 
	    //		    << std::endl;
	    if( capid != capid2 || adc !=  adc2) {
	      std::cout << "===> PROBLEM !!!  gebsubdet=" << gsub 
			<< " zside=" <<zside
			<< " section= "<< section << " channel " <<channel
			<< std::endl;
	      std::cout << "     capid1["<< i << "]=" << capid 
			<< " adc1["<< i << "]=" << adc 
			<< "     capid2["<< i << "]=" << capid2 
			<< " adc2["<< i << "]=" << adc2
			<< std::endl;   
	      identical = 0;
	      meStatus->Fill(1.); 
	      break;
	    }
	  } // end of DataFrames array
	  if(identical) meStatus->Fill(0.); 
	  break; // matched HcalZDCID  is processed,
	  //  go to next (primary collection) cell  
	} 
      } // end of cycle over 2d DIGI collection 
      if (!match) {
	meStatus->Fill(2.); 
	std::cout << "===> PROBLEM !!!  gsubdet=" << gsub
		  << " zside=" <<zside
		  << " section= "<< section << " channel " <<channel
		  << " HcalZDCId match is not found !!!" 
		  << std::endl;
      }
    }
    else{
      //for Hcal subdetectors
      HcalDetId cell(digiItr1->id()); 
      int depth = cell.depth();
      int iphi  = cell.iphi()-1;
      int ieta  = cell.ieta();
      int sub   = cell.subdet();
      //    if(ieta > 0) ieta--;
      //    std::cout << " Cell subdet=" << sub << "  ieta=" << ieta 
      //	      << "  inphi=" << iphi << "  depth=" << depth << std::endl;
      
      // CYCLE over second DIGI collection ======================================
      for (digiItr2=digiCollection2->begin();digiItr2!=digiCollection2->end();digiItr2++) {
  
	HcalDetId cell2(digiItr2->id());
	
	if( cell == cell2) {
	  match = 1;
	  int identical = 1; 
	  for (int i=0; i<tsize; i++) {
	    double adc  =  (*digiItr1)[i].adc();     
	    int capid   =  (*digiItr1)[i].capid();
	    //	  std::cout << std::endl << "     capid1=" << capid 
	    //                << " adc1=" << adc 
	    //		    << std::endl;
	    double adc2 =  (*digiItr2)[i].adc();     
	    int capid2  =  (*digiItr2)[i].capid();
	    //	  std::cout << "     capid2=" << capid2 << " adc2=" << adc2 
	    //		    << std::endl;
	    if( capid != capid2 || adc !=  adc2) {
	      std::cout << "===> PROBLEM !!!  subdet=" << sub << "  ieta="
			<< ieta  << "  inphi=" << iphi << "  depth=" << depth
			<< std::endl;
	      std::cout << "     capid1["<< i << "]=" << capid 
			<< " adc1["<< i << "]=" << adc 
			<< "     capid2["<< i << "]=" << capid2 
			<< " adc2["<< i << "]=" << adc2
			<< std::endl;   
            identical = 0;
            meStatus->Fill(1.); 
            break;
	    }
	  } // end of DataFrames array
	  if(identical) meStatus->Fill(0.); 
	  break; // matched HcalID  is processed,
	  //  go to next (primary collection) cell  
	} 
      } // end of cycle over 2d DIGI collection 
      if (!match) {
	meStatus->Fill(2.); 
	std::cout << "===> PROBLEM !!!  subdet=" << sub << "  ieta="
		  << ieta  << "  inphi=" << iphi << "  depth=" << depth
		  << " HcalID match is not found !!!" 
		  << std::endl;
      }
    }
  } // end of cycle over 1st DIGI collection    
  
  if (size1 != size2) {
    meStatus->Fill(3.); 
    std::cout << "===> PROBLEM !!!  Different size of Digi collections : "
	      << size1  << " and " << size2 
	      << std::endl;
  }
}


Digi2Raw2Digi::Digi2Raw2Digi(const edm::ParameterSet& iConfig)
  : inputTag1_(iConfig.getParameter<edm::InputTag>("digiLabel1")),
    inputTag2_(iConfig.getParameter<edm::InputTag>("digiLabel2")),
    outputFile_(iConfig.getUntrackedParameter<std::string>("outputFile")),
    dbe_(0)
{ 

  // DQM ROOT output
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo")
      << " Hcal RecHit Task histograms will be saved to '" 
      << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") 
      << " Hcal RecHit Task histograms will NOT be saved";
  }

  // DQM service initialization
  dbe_ = edm::Service<DQMStore>().operator->();
   
  if ( dbe_ ) {
    dbe_->setCurrentFolder("Digi2Raw2DigiV/Digi2Raw2DigiTask");
  }

  // const char * sub = hcalselector_.c_str();
  char histo[100];
  sprintf (histo, "Digi2Raw2Digi_status") ;
  // bins: 1)full match 2)ID match, not content 3) no match 
  // 4) number of events with diff number of Digis
  meStatus    = dbe_->book1D(histo, histo, 5, 0., 5.);

}


Digi2Raw2Digi::~Digi2Raw2Digi() { 

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}


void Digi2Raw2Digi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  unsuppressed = 0;
  
  //  std::cout << "=== HBHE ==================" << std::endl; 
  compare<HBHEDataFrame>(iEvent,iSetup);

  //  std::cout << "=== HO ====================" << std::endl; 
  compare<HODataFrame>(iEvent,iSetup);

  //  std::cout << "=== HF ====================" << std::endl; 
  compare<HFDataFrame>(iEvent,iSetup);  


  //  std::cout << "=== ZDC ===================" << std::endl; 
  unsuppressed = 1;
  compare<ZDCDataFrame>(iEvent,iSetup);
  
  
  //  std::cout << "=== CASTOR ================" << std::endl; 
  //  compare<CastorDataFrame>(iEvent,iSetup); 
  
}

DEFINE_SEAL_MODULE ();
DEFINE_ANOTHER_FWK_MODULE (Digi2Raw2Digi) ;
