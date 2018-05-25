/** \class EcalEBTrigPrimClusterAlgo
 *
 * EcalEBTrigPrimClusterAlgo 
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
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTrigPrimClusterAlgo.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"


EcalEBTrigPrimClusterAlgo::EcalEBTrigPrimClusterAlgo(const edm::EventSetup & setup,int nSam, int binofmax,bool tcpFormat, bool barrelOnly,bool debug, bool famos)

{
  barrelOnly_=barrelOnly;
  binOfMaximum_=binofmax;
  tcpFormat_=tcpFormat;
  debug_=debug; 
  famos_=famos;
  nSamples_=nSam;
  maxNrSamples_=10;
  this->init(setup);

  std::vector <float> etVec;
  etVec.resize(maxNrSamples_);
  for (int i=0;i<maxNrSamples_;i++) etVec[i]=0.;
  clusters_out_.resize(10000);  
  for (int i=0;i<10000;i++) clusters_out_[i]=etVec;

  fenixTcpFormatClu_ = new EcalFenixTcpFormatCluster(tcpFormat_, debug_, famos_, binOfMaximum_);
  tcpformat_out_.resize(maxNrSamples_);   


}



EcalEBTrigPrimClusterAlgo::~EcalEBTrigPrimClusterAlgo() 
{
  for (int i=0;i<nbMaxXtals_;i++) delete linearizer_[i]; 
  delete amplitude_filter_;
  delete peak_finder_;
  delete fenixFormatterEB_;
  delete fenixTcpFormatClu_;
}


void EcalEBTrigPrimClusterAlgo::run(const edm::EventSetup & setup, 
				    EBDigiCollection const * digi,
				    EcalEBClusterTrigPrimDigiCollection & result,
				    EcalEBClusterTrigPrimDigiCollection & resultTcp, 
				    int dEta, int dPhi, 
				    double hitNoiseCut, double eCutOnSeed)
{

  //typedef typename Coll::Digi Digi;
  if (debug_) {
    std::cout << "  EcalEBTrigPrimClusterAlgo: Testing that the algorythm with digis is well plugged " << std::endl;
    std::cout << "  EcalEBTrigPrimClusterAlgo: digi size " << digi->size() << std::endl;
  }


 edm::ESHandle<CaloSubdetectorGeometry> theBarrelGeometry_handle;
  setup.get<EcalBarrelGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);
  const CaloSubdetectorGeometry *theBarrelGeometry;
  theBarrelGeometry = &(*theBarrelGeometry_handle);
  

  firstSample_ = binOfMaximum_-1 -nrSamples_/2;
  lastSample_ = binOfMaximum_-1 +nrSamples_/2;

  if (debug_) {
    std::cout << "  binOfMaximum_ " <<  binOfMaximum_ << " nrSamples_" << nrSamples_ << std::endl;
    std::cout << " first sample " << firstSample_ << " last " << lastSample_ <<std::endl;
  }

  std::vector<std::vector<SimpleCaloHit> > hitCollection;   
  uint16_t etInADC;

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

	  std::vector<SimpleCaloHit> singleChan;
	  for (int iSample=firstSample_;iSample<=lastSample_;++iSample) {	
            SimpleCaloHit singleSample(0);
	    etInADC= tcpformat_out_[iSample];
	    singleSample.setEtInADC(etInADC);
            singleSample.setId(myid);
            singleSample.setPosition(GlobalVector( theBarrelGeometry->getGeometry(myid)->getPosition().x(), 
						   theBarrelGeometry->getGeometry(myid)->getPosition().y(), 
						   theBarrelGeometry->getGeometry(myid)->getPosition().z()));
	    //	    std::cout << " iSample " << iSample << " et ADC " << etInADC << " ID " << myid << " position " << singleSample.position() << std::endl;	    

	    singleChan.push_back(singleSample);
	  }
	  hitCollection.push_back(singleChan); 
	  

	} // Loop over the xStals in  a strip

      }//loop over strips in one tower

  }   // loop over the towers 

  
  std::vector<uint16_t> cluCollection = makeCluster ( hitCollection, result, dEta, dPhi, hitNoiseCut, eCutOnSeed  );
  if (debug_) std::cout << "  TrigPrimClusterAlgo hitCollection size " << hitCollection.size() << " cluster size " << cluCollection.size() <<  std::endl; 


}



std::vector<uint16_t>  EcalEBTrigPrimClusterAlgo::makeCluster ( std::vector<std::vector<SimpleCaloHit> >&  hitCollection, 
								EcalEBClusterTrigPrimDigiCollection & result, 
								int dEta, int dPhi,
                                                                double hitNoiseCut,
								double eCutOnSeed) {

  EcalEBClusterTriggerPrimitiveDigi tp;

   if (debug_) std::cout << "  makeCluster  input collection size " << hitCollection.size() << std::endl;
   std::vector<std::vector<uint16_t> > clusters;
   std::vector<std::vector<float> >::const_iterator iClu;


   std::vector<uint16_t> etVec;
   
   int iSample=2;
   

   while (true) {
     
     SimpleCaloHit centerhit(0);      
     
     for (unsigned int iChan=0;iChan<hitCollection.size();++iChan) {
       SimpleCaloHit  hit = hitCollection[iChan][iSample];  
       
       
       float energy = hit.energy();
       if ( energy < hitNoiseCut ) continue;
       
       if ( !hit.stale && hit.etInGeV() > centerhit.etInGeV() ) {
	 centerhit = hit;  
	 
       }      
       if (debug_) std::cout << "  makeCluster energy " << energy << " " << hit.energy() << " et " << hit.etInGeV() << " stale " << hit.stale << std::endl;
       
       
     } //looping over the pseudo-hits
     
     if ( centerhit.energy() <= eCutOnSeed ) break;
     centerhit.stale=true;
     if (debug_) {
       std::cout << "-------------------------------------" << std::endl;
       std::cout << "New cluster: center crystal pt = " << centerhit.etInGeV() << std::endl;
     }
     
     
     
     GlobalVector weightedPosition;
     float totalEnergy = 0.;
     std::vector<float> crystalEt;
     std::vector<EBDetId> crystalId;
     
     for (unsigned int iChan=0;iChan<hitCollection.size();++iChan) {
       SimpleCaloHit &hit(hitCollection[iChan][iSample]);  

       float energy = hit.energy();
       if ( energy < hitNoiseCut ) continue;
       
       if ( !hit.stale &&  (abs(hit.dieta(centerhit)) < dEta && abs(hit.diphi(centerhit)) <  dPhi ) ) {
	 
	 weightedPosition += hit.position()*hit.energy();
	 if (debug_) std::cout << " evolving  weightedPosition " << weightedPosition.eta() << " " << weightedPosition.phi() << std::endl;
	 totalEnergy += hit.energy();
	 hit.stale = true;
	 crystalEt.push_back(hit.etInGeV());
	 crystalId.push_back(hit.id());
	 
	 
	 if ( debug_) {
	   if ( hit == centerhit )
	     std::cout << "      "; 
	   std::cout <<
	     "\tCrystal (" << hit.dieta(centerhit) << "," << hit.diphi(centerhit) <<
	     ") Et in ADC =" << hit.etInADC() <<
	     ", Et in GeV =" << hit.etInGeV() <<
	     ", eta=" << hit.position().eta() <<
	     ", phi=" << hit.position().phi() << std::endl;
	 }
       }
     }
     float totalEt = totalEnergy*sin(weightedPosition.theta());
     uint16_t etInADC =  totalEt/0.125;
     float etaClu= weightedPosition.eta() ;
     float phiClu= weightedPosition.phi();
     if (debug_) std::cout << " Cluster totalenergy " << totalEnergy << " total Et " << totalEt << " total Et in ADC " << etInADC << " weighted eta " << etaClu << " weighted phi " <<  phiClu << std::endl;
     
     tp=EcalEBClusterTriggerPrimitiveDigi(centerhit.id(), crystalId, etaClu,  phiClu );   
     tp.setSize( nrSamples_);
     
     
     etVec.push_back(etInADC);
     clusters.push_back(etVec);
     
     bool isASpike=0; 
     int timing=0;
     tp.setSample(2,  EcalEBClusterTriggerPrimitiveSample(etInADC,isASpike,timing)  ); 
     result.push_back(tp);
     
     
   } // while true 
   
   
   
   if (debug_) {
     std::cout << " etVec size " << etVec.size() <<  " Clusters size " << clusters.size() << std::endl;  
     for (unsigned int iet=0; iet < etVec.size(); iet++) std::cout << " Et vec " << etVec[iet] << " ";
     std::cout << " " << std::endl;
   }
   
   
   return etVec;
   
}

  


