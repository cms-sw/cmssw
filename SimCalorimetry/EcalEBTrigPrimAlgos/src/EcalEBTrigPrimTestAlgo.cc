/** \class EcalEBTrigPrimTestAlgo
 *
 * EcalEBTrigPrimTestAlgo 
 * starting point for Phase II: build TPs out of Phase I digis to start building the
 * infrastructures
 *
 *
 ************************************************************/
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
//#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTrigPrimTestAlgo.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"



#include <TTree.h>
#include <TMath.h>


//----------------------------------------------------------------------

const unsigned int EcalEBTrigPrimTestAlgo::nrSamples_=5;
const unsigned int EcalEBTrigPrimTestAlgo::maxNrTowers_=2448;
const unsigned int EcalEBTrigPrimTestAlgo::maxNrSamplesOut_=10;


EcalEBTrigPrimTestAlgo::EcalEBTrigPrimTestAlgo(const edm::EventSetup & setup,int nSam, int binofmax,bool tcpFormat, bool barrelOnly,bool debug, bool famos): 
  nSamples_(nSam),binOfMaximum_(binofmax), tcpFormat_(tcpFormat), barrelOnly_(barrelOnly), debug_(debug), famos_(famos)

{

 maxNrSamples_=10;
 this->init(setup);
}

//----------------------------------------------------------------------
void EcalEBTrigPrimTestAlgo::init(const edm::EventSetup & setup) {
  if (!barrelOnly_) {
    //edm::ESHandle<CaloGeometry> theGeometry;
    //    edm::ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle;
    setup.get<CaloGeometryRecord>().get( theGeometry );
    setup.get<IdealGeometryRecord>().get(eTTmap_);
  }

  // initialise data structures
  initStructures(towerMapEB_);
  hitTowers_.resize(maxNrTowers_);


  linearizer_.resize(nbMaxXtals_);
  for (int i=0;i<nbMaxXtals_;i++) linearizer_[i] = new  EcalFenixLinearizer(famos_);

  //
  std::vector <int> v;
  v.resize(maxNrSamples_);
  lin_out_.resize(nbMaxXtals_);  
  for (int i=0;i<5;i++) lin_out_[i]=v;
  //
  amplitude_filter_ = new EcalFenixAmplitudeFilter();
  filt_out_.resize(maxNrSamples_);
  peak_out_.resize(maxNrSamples_);
  // these two are dummy
  fgvb_out_.resize(maxNrSamples_);
  fgvb_out_temp_.resize(maxNrSamples_);  
  //
  peak_finder_ = new  EcalFenixPeakFinder();
  fenixFormatterEB_ = new EcalFenixStripFormatEB();
  format_out_.resize(maxNrSamples_);
  //
  fenixTcpFormat_ = new EcalFenixTcpFormat(tcpFormat_, debug_, famos_, binOfMaximum_);
  tcpformat_out_.resize(maxNrSamples_);   

}
//----------------------------------------------------------------------

EcalEBTrigPrimTestAlgo::~EcalEBTrigPrimTestAlgo() 
{
  for (int i=0;i<nbMaxXtals_;i++) delete linearizer_[i]; 
  delete amplitude_filter_;
  delete peak_finder_;
  delete fenixFormatterEB_;
  delete fenixTcpFormat_;
}

/*
void EcalEBTrigPrimTestAlgo::run(const edm::EventSetup & setup, EcalRecHitCollection const * rh,
				 EcalEBTrigPrimDigiCollection & result,
				 EcalEBTrigPrimDigiCollection & resultTcp)
{



  //std::cout << "  EcalEBTrigPrimTestAlgo: Testing that the algorythm is well plugged " << std::endl;
  //std::cout << "  EcalEBTrigPrimTestAlgo: recHit size " << rh->size() << std::endl;

  edm::ESHandle<CaloSubdetectorGeometry> theBarrelGeometry_handle;
  setup.get<EcalBarrelGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);
  const CaloSubdetectorGeometry *theBarrelGeometry;
  theBarrelGeometry = &(*theBarrelGeometry_handle);

  
  EcalEBTriggerPrimitiveDigi tp;
  std::vector<EcalTriggerPrimitiveSample> tpSam[10];


  uint16_t fEt;
  double EtSat=128.;

  for (unsigned int i=0;i<rh->size();i++) {
    const EBDetId & myid1=(*rh)[i].id();
    if ( (*rh)[i].energy() < 0.2 ) continue;
    tp=  EcalEBTriggerPrimitiveDigi(  myid1);   
    tp.setSize(nSamples_);
    int nSam=0;

    for (int iSample=0; iSample<nSamples_; iSample++) {
      if (debug_)  std::cout << " DetId " << myid1 << "Subdetector " << myid1.subdet() << " ieta " << myid1.ieta() << " iphi " << myid1.iphi() << std::endl;
      
      float theta =  theBarrelGeometry->getGeometry(myid1)->getPosition().theta();
      //uint16_t et=((*rh)[i].energy())*sin(theta);
      double et=((*rh)[i].energy())*sin(theta);

      double lsb10bits = 0. ;
      int tpgADC10b = 0.;
      lsb10bits = EtSat/1024. ;
      if (lsb10bits>0) 
	tpgADC10b = int(et/lsb10bits+0.5) ;
      
      if (debug_) std::cout << " Et in GeV " << et << " tpgADC10b " << tpgADC10b << std::endl;
      fEt=et;

     
      // if (fEt >0xfff)
      // 	fEt=0xfff;
      //fEt >>=2;
      //if (fEt>0x3ff) fEt=0x3ff;
     
      //std::cout << " Et after formatting " << fEt << std::endl;
      EcalTriggerPrimitiveSample mysam(fEt);
      tp.setSample(nSam, mysam );
      nSam++;
      if (debug_) std::cout << "in TestAlgo" <<" tp size "<<tp.size() << std::endl;
    }
 
    

    if (!tcpFormat_)
      result.push_back(tp);
    else 
      resultTcp.push_back(tp);

   
    if (debug_) std::cout << " result size " << result.size() << std::endl;

  }

}
*/


void EcalEBTrigPrimTestAlgo::run(const edm::EventSetup & setup, 
				 EBDigiCollection const * digi,
				 EcalEBTrigPrimDigiCollection & result,
				 EcalEBTrigPrimDigiCollection & resultTcp)
{

  //typedef typename Coll::Digi Digi;
  if (debug_) {
    std::cout << "  EcalEBTrigPrimTestAlgo: Testing that the algorythm with digis is well plugged " << std::endl;
    std::cout << "  EcalEBTrigPrimTestAlgo: digi size " << digi->size() << std::endl;
  }

  uint16_t etInADC;
  EcalEBTriggerPrimitiveDigi tp;
  int firstSample = binOfMaximum_-1 -nrSamples_/2;
  int lastSample = binOfMaximum_-1 +nrSamples_/2;

  if (debug_) {
    std::cout << "  binOfMaximum_ " <<  binOfMaximum_ << " nrSamples_" << nrSamples_ << std::endl;
    std::cout << " first sample " << firstSample << " last " << lastSample <<std::endl;
  }

  clean(towerMapEB_);
  fillMap(digi,towerMapEB_);

  for(int itow=0;itow<nrTowers_;++itow)  {

    int index=hitTowers_[itow].first;
    const EcalTrigTowerDetId &thisTower=hitTowers_[itow].second;
    if (debug_) std::cout << " Data for TOWER num " << itow << " index " << index << " TowerId " << thisTower <<  " size " << towerMapEB_[itow].size() << std::endl;    
    // loop over all strips assigned to this trigger tower
    int nxstals=0;
    for(unsigned int iStrip = 0; iStrip < towerMapEB_[itow].size();++iStrip)
      {
	if (debug_) std::cout << " Data for STRIP num " << iStrip << std::endl;    
	std::vector<EBDataFrame> &dataFrames = (towerMapEB_[index])[iStrip].second;//vector of dataframes for this strip, size; nr of crystals/strip

	nxstals = (towerMapEB_[index])[iStrip].first;
	if (nxstals <= 0) continue;
	if (debug_) std::cout << " Number of xTals " << nxstals << std::endl;
	
	const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(dataFrames[0].id());
	uint32_t stripid=elId.rawId() & 0xfffffff8;   


	// loop over the xstals in a strip
	for (int iXstal=0;iXstal<nxstals;iXstal++) {
	  const EBDetId & myid = dataFrames[iXstal].id();
	  tp=  EcalEBTriggerPrimitiveDigi(  myid );   
	  tp.setSize( nrSamples_);


	  if(debug_){
	    std::cout<<std::endl;
	    std::cout <<"iXstal= "<<iXstal<< " id " <<  dataFrames[iXstal].id()  << " EBDataFrame is: "<<std::endl; 
	    for ( int i = 0; i<dataFrames[iXstal].size();i++){
	      std::cout <<" "<<std::dec<<dataFrames[iXstal][i].adc();
	    }
	    std::cout<<std::endl;
	  }
	  //   Call the linearizer
	  this->getLinearizer(iXstal)->setParameters( dataFrames[iXstal].id().rawId(),ecaltpPed_,ecaltpLin_,ecaltpgBadX_) ; 
	  this->getLinearizer(iXstal)->process( dataFrames[iXstal],lin_out_[iXstal]);

	  for (unsigned int i =0; i<lin_out_[iXstal].size();i++){
	    if( (lin_out_[iXstal])[i]>0X3FFFF) (lin_out_[iXstal])[i]=0X3FFFF;
	  }

 
          if ( debug_ ) {
	    std::cout<< "output of linearizer for channel " << iXstal << std::endl; 
	    std::cout<<" lin_out[iXstal].size()= "<<std::dec<<lin_out_[iXstal].size()<<std::endl;
	    for (unsigned int i =0; i<lin_out_[iXstal].size();i++){
	      std::cout <<" "<<std::dec<<(lin_out_[iXstal])[i];
	    }
	    std::cout<<std::endl;
	  }



	  // Call the amplitude filter
	  this->getFilter()->setParameters(stripid,ecaltpgWeightMap_,ecaltpgWeightGroup_);      
	  this->getFilter()->process(lin_out_[iXstal],filt_out_,fgvb_out_temp_,fgvb_out_);   

	  if(debug_){
	    std::cout<< "output of filter is a vector of size: "<<std::dec<<filt_out_.size()<<std::endl; 
	    for (unsigned int ix=0;ix<filt_out_.size();ix++){
	      std::cout<<std::dec<<filt_out_[ix] << " " ;
	    }
	    std::cout<<std::endl;
	  }

	  // call peakfinder
	  this->getPeakFinder()->process(filt_out_,peak_out_);
 
	  if(debug_){
	    std::cout<< "output of peakfinder is a vector of size: "<<std::dec<<peak_out_.size()<<std::endl; 
	    for (unsigned int ix=0;ix<peak_out_.size();ix++){
	      std::cout<<std::dec<<peak_out_[ix] << " " ;
	    }
	    std::cout<<std::endl;
	  }

	  // call formatter
	  this->getFormatterEB()->setParameters(stripid,ecaltpgSlidW_) ; 
	  this->getFormatterEB()->process(fgvb_out_,peak_out_,filt_out_,format_out_); 

	  if (debug_) {
	    std::cout<< "output of formatter is a vector of size: "<<format_out_.size()<<std::endl; 
	    for (unsigned int i =0; i<format_out_.size();i++){
	      std::cout <<" "<<std::dec<<format_out_[i] << " " ;
	    }    
	    std::cout<<std::endl;
	  }
	  

	  // call final tcp formatter
	  this->getFormatter()->setParameters( thisTower.rawId(),ecaltpgLutGroup_,ecaltpgLut_,ecaltpgBadTT_,ecaltpgSpike_);
	  this->getFormatter()->process(format_out_,tcpformat_out_);

	  // loop over the time samples and fill the TP
	  int nSam=0;
	  for (int iSample=firstSample;iSample<=lastSample;++iSample) {
	    etInADC= tcpformat_out_[iSample];
	    if (debug_) std::cout << " format_out " << tcpformat_out_[iSample] <<  " etInADC " << etInADC << std::endl;
	    //	    EcalTriggerPrimitiveSample mysam(etInADC);
	    //tp.setSample(nSam, mysam );

	    tp.setSample(nSam,  EcalTriggerPrimitiveSample(etInADC, false, 0)  );

	    nSam++;
	    if (debug_) std::cout << "in TestAlgo" <<" tp size "<<tp.size() << std::endl;
	  }



	  if (!tcpFormat_)
	    result.push_back(tp);
	  else 
	    resultTcp.push_back(tp);
	  


	} // Loop over the xStals




      }//loop over strips in one tower
  



  }




  /*
  for (unsigned int i=0;i<digi->size();i++) {
    EBDataFrame myFrame((*digi)[i]);  
    const EBDetId & myid1 = myFrame.id();
    tp=  EcalTriggerPrimitiveDigi(  myid1);   
    tp.setSize( myFrame.size());
    int nSam=0;

    if (debug_) {
      std::cout << " data frame size " << myFrame.size() << " Id " <<  myFrame.id()  << std::endl;
      std::cout << " Sample data ADC: " << std::endl;
      for (int iSample=0; iSample<myFrame.size(); iSample++) {
	std::cout << " " << std::dec<< myFrame.sample(iSample).adc() ;
      }
      std::cout<<std::endl;
    }

    
    this->getLinearizer(i)->setParameters( myFrame.id().rawId(),ecaltpPed_,ecaltpLin_,ecaltpgBadX_) ; 
    //this->getLinearizer(i)->process( myFrame,lin_out_[i]);

    if (debug_) {
      std::cout<< "cryst: "<< i <<"  value : "<<std::dec<<std::endl;
      std::cout<<" lin_out[i].size()= "<<std::dec<<lin_out_[i].size()<<std::endl;
      for (unsigned int j =0; j<lin_out_[i].size();j++){
	std::cout <<" "<<std::dec<<(lin_out_[i])[j];
      }
      std::cout<<std::endl;
    }


    for (int iSample=0; iSample<myFrame.size(); iSample++) {
      etInADC= myFrame.sample(iSample).adc();
      EcalTriggerPrimitiveSample mysam(etInADC);
      tp.setSample(nSam, mysam );
      nSam++;
      if (debug_) std::cout << "in TestAlgo" <<" tp size "<<tp.size() << std::endl;
    }

    if (!tcpFormat_)
      result.push_back(tp);
    else 
      resultTcp.push_back(tp);
    
   
    if (debug_) std::cout << " result size " << result.size() << std::endl;
    
    
    
  }
  */

}


  



//----------------------------------------------------------------------

int  EcalEBTrigPrimTestAlgo::findStripNr(const EBDetId &id){
  int stripnr;
  int n=((id.ic()-1)%100)/20; //20 corresponds to 4 * ecal_barrel_crystals_per_strip FIXME!!
  if (id.ieta()<0) stripnr = n+1;
  else stripnr =nbMaxStrips_ - n; 
  return stripnr;
}

