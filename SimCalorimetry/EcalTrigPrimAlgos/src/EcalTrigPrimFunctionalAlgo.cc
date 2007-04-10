/** \class EcalTrigPrimFunctionalAlgo
 *
 * EcalTrigPrimFunctionalAlgo is the main algorithm class for TPG
 * It coordinates all the other algorithms
 * Structure is very close to electronics
 *
 *
 * \author Ursula Berthon, Stephanie Baffioni,  LLR Palaiseau
 *
 * \version   1st Version may 2006
 * \version   2nd Version jul 2006

 *
 ************************************************************/
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalTrigPrimFunctionalAlgo.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h"
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"
#include "CondFormats/DataRecord/interface/EcalTPParametersRcd.h"

#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include <TTree.h>
#include <TMath.h>
//----------------------------------------------------------------------

EcalTrigPrimFunctionalAlgo::EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup,int binofmax,int nrsamples, bool tcpFormat, bool barrelOnly,bool debug, double ebDccAdcToGeV,double eeDccAdcToGeV):
  valid_(false),valTree_(NULL),binOfMaximum_(binofmax),nrSamplesToWrite_(nrsamples),
  tcpFormat_(tcpFormat), barrelOnly_(barrelOnly), debug_(debug),
  ebDccAdcToGeV_(ebDccAdcToGeV),eeDccAdcToGeV_(eeDccAdcToGeV)
{this->init(setup);}

//----------------------------------------------------------------------
EcalTrigPrimFunctionalAlgo::EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup,TTree *tree,int binofmax, int nrsamples,bool tcpFormat, bool barrelOnly, bool debug, double ebDccAdcToGeV,double eeDccAdcToGeV):
  valid_(true),valTree_(tree),binOfMaximum_(binofmax),nrSamplesToWrite_(nrsamples),
  tcpFormat_(tcpFormat), barrelOnly_(barrelOnly),debug_(debug),
  ebDccAdcToGeV_(ebDccAdcToGeV),eeDccAdcToGeV_(eeDccAdcToGeV)
{this->init(setup);}

//----------------------------------------------------------------------
void EcalTrigPrimFunctionalAlgo::init(const edm::EventSetup & setup) {
  if (!barrelOnly_) {
    edm::ESHandle<CaloGeometry> theGeometry;
    edm::ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle;
    setup.get<IdealGeometryRecord>().get( theGeometry );
    setup.get<IdealGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
    theEndcapGeometry = &(*theEndcapGeometry_handle);
    setup.get<IdealGeometryRecord>().get(eTTmap_);
  }
  edm::ESHandle<EcalTPParameters> theEcalTPParameters_handle;
  setup.get<EcalTPParametersRcd>().get(theEcalTPParameters_handle);
  ecaltpp_=theEcalTPParameters_handle.product();

  ebstrip_= new EcalBarrelFenixStrip(valTree_,ecaltpp_,debug_);
  ebtcp_ = new EcalBarrelFenixTcp(ecaltpp_,tcpFormat_,debug_) ;
}
//----------------------------------------------------------------------
void EcalTrigPrimFunctionalAlgo::updateESRecord(double ttfLowEB, double ttfHighEB, double ttfLowEE, double ttfHighEE)
{
  ecaltpp_->changeThresholds(ttfLowEB, ttfHighEB, ttfLowEE, ttfHighEE);
}
//----------------------------------------------------------------------
EcalTrigPrimFunctionalAlgo::~EcalTrigPrimFunctionalAlgo() 
{
    delete ebstrip_;
    delete ebtcp_;
}

//----------------------------------------------------------------------

void EcalTrigPrimFunctionalAlgo::run(const EBDigiCollection* ebdcol,const EEDigiCollection* eedcol, EcalTrigPrimDigiCollection & result, EcalTrigPrimDigiCollection & resultTcp) {
  
  sumBarrel_.clear();
 
  int nhitsb(0), nhitse(0);
  
//   static SimpleConfigurable<float> ratBL(0.8,"ECALBarrel:FGLowEnRatioTh");
//   static SimpleConfigurable<float> ratBH(0.9,"ECALBarrel:FGHighEnRatioTh");
//   static SimpleConfigurable<float> enBL(5.0,"ECALBarrel:FGLowEnTh");
//   static SimpleConfigurable<float> enBH(25.0,"ECALBarrel:FGHighEnTh");
  
//   //SimpleConfigurable<float> ratEL(0.8,"ECALEndcap:FGLowEnRatioTh");
//   //SimpleConfigurable<float> ratEH(0.9,"ECALEndcap:FGHighEnRatioTh");
//   static SimpleConfigurable<float> enEL(5.0,"ECALEndcap:FGLowEnTh");
//   static SimpleConfigurable<float> enEH(25.0,"ECALEndcap:FGHighEnTh");


  
// loop over dataframes and fill map for barrel
  if (ebdcol) {
    for(unsigned int i = 0; i < ebdcol->size() ; ++i) {
      const EBDetId & myid=(*ebdcol)[i].id();
      const EcalTrigTowerDetId coarser= myid.tower();
      if(coarser.null())  
	{
	  LogDebug("EcalTPG")<< "Cell " << myid << " has trivial coarser granularity (probably EFRY corner, not in this tower map; hit ignored)";
	  continue;
	}	
	
      nhitsb++;
      fillBarrel(coarser,(*ebdcol)[i]);
    }// loop over all CaloDataFrames
    LogDebug("EcalTPG")<< "[EcalTrigPrimFunctionalAlgo] (found " << nhitsb << " frames in " 
		<< sumBarrel_.size() << " Barrel towers  ";
  }

  // loop over dataframes and fill map for endcap
  if (!barrelOnly_) {
    mapEndcap_.clear();
    if (eedcol) {
      for(unsigned int i = 0; i < eedcol->size() ; ++i) {
	const EEDetId & myid=(*eedcol)[i].id();
	EcalTrigTowerDetId coarser=(*eTTmap_).towerOf(myid);

	nhitse++;
	fillEndcap(coarser,(*eedcol)[i]);

      }// loop over all EEDataFrames
      LogDebug("EcalTPG") << "[EcalTrigPrimFunctionalAlgo] (found " << nhitse << " frames in " 
		   << mapEndcap_.size() << " Endcap towers  ";
    }
  }
  // prepare writing of TP-s

  int firstSample = binOfMaximum_-1 -nrSamplesToWrite_/2;
  int lastSample = binOfMaximum_-1 +nrSamplesToWrite_/2;
 

  //   Barrel treatment

  SUMVB::const_iterator it = sumBarrel_.begin(); 
  SUMVB::const_iterator e = sumBarrel_.end(); 

  int itow=0;
  // loop over all trigger towers
  for(;it!=e;it++) 
    {
      itow++;
      const EcalTrigTowerDetId & thisTower =(*it).first;
      int townr = findTowerNrInSM ( thisTower);
      // loop over all strips assigned to this trigger tower
      std::vector<std::vector<int> > striptp;
      for(unsigned int i = 0; i < TMath::Min(it->second.size(),size_t(ecal_barrel_strips_per_trigger_tower)) ; ++i) 
	{
	  std::vector<int> tp;
	  std::vector<const EBDataFrame *> df=it->second[i];

	  if (df.size()>0) {
	    tp=ebstrip_->process(df,i+1,townr);
	    striptp.push_back(tp);
	  }
	}


      std::vector<EcalTriggerPrimitiveSample> towtp;
      std::vector<EcalTriggerPrimitiveSample> towtp2;
      ebtcp_->process(striptp,towtp, towtp2, 1, townr);  //PP 1 should be Supermodule nb

      // Fill TriggerPrimitiveDigi
      EcalTriggerPrimitiveDigi tptow(thisTower);
      tptow.setSize(nrSamplesToWrite_);
      if (towtp.size()<nrSamplesToWrite_)  { 
	edm::LogWarning("Barrel") <<"Too few samples produced, nr is "<<towtp.size();
	break;
      }
      int isam=0;
      for (int i=firstSample;i<=lastSample;++i) {
	tptow.setSample(isam++,EcalTriggerPrimitiveSample(towtp[i]));
      }
      //      LogDebug("EcalTPG") <<"For "<<thisTower<<" the following TP was created: "<<towtp[i];
      result.push_back(tptow);
      if (tcpFormat_) {
	EcalTriggerPrimitiveDigi tptow(thisTower);
	tptow.setSize(nrSamplesToWrite_);
	if (towtp2.size()<nrSamplesToWrite_)  { 
	  edm::LogWarning("Barrel") <<"Too few samples produced, nr is "<<towtp2.size();
	  break;
	}
	int isam=0;
	for (int i=firstSample;i<=lastSample;++i) {
	  tptow.setSample(isam++,EcalTriggerPrimitiveSample(towtp2[i]));
	}
	resultTcp.push_back(tptow);
      }
    }

  //   Endcap treatment
  // completely temporary, waiting for endcap geometry !!!

  if (!barrelOnly_) {
    MAPE::const_iterator ite = mapEndcap_.begin(); 
    MAPE::const_iterator ee = mapEndcap_.end(); 

	int one=0,two=0;

    itow=0;
    // loop over all trigger towers
    for(;ite!=ee;ite++) 
      {
	itow++;
	const EcalTrigTowerDetId & thisTower =(*ite).first;
	int nrFrames=mapEndcap_[thisTower].size();

	// first, calculate thresholds
	std::vector<int>  thresholds(nrFrames);
	for (int ii=0;ii<nrFrames;++ii) {
	  thresholds[ii]=((mapEndcap_[thisTower][ii])[0].adc()+(mapEndcap_[thisTower][ii])[1].adc()+(mapEndcap_[thisTower][ii])[2].adc())/3;
	}

	std::vector<EcalTriggerPrimitiveDigi> tptow;
	// special treatment for the 2 inner rings: 2 pseudo-towers for one physical tower
	int nrTowers;
        if (thisTower.ietaAbs()==27 | thisTower.ietaAbs()==28 ) {
	  //special treatment for 2 inner eta rings
	  nrTowers=2;
	  int phi=2*((thisTower.iphi()-1)/2);
	  tptow.push_back(EcalTriggerPrimitiveDigi(EcalTrigTowerDetId(thisTower.zside(),thisTower.subDet(),thisTower.ietaAbs(),phi+1)));
	  tptow.push_back(EcalTriggerPrimitiveDigi(EcalTrigTowerDetId(thisTower.zside(),thisTower.subDet(),thisTower.ietaAbs(),phi+2)));
	  two++;
	} else {
	  one++;
	  nrTowers=1;
	  tptow.push_back(EcalTriggerPrimitiveDigi(thisTower));
	}

        // fill TP-s for each sample
	unsigned int nrSamples=mapEndcap_[thisTower][0].size();
	if (nrSamples<nrSamplesToWrite_)  { //UB FIXME: exception?
	  edm::LogWarning("Endcap") <<"Too few samples produced, nr is "<<nrSamples;
	  break;
	}
	// calculate Et and rescale it to correspond to barrel values
	// as long as we dont have correct parameters for the endcap
	std::vector<EcalTriggerPrimitiveSample> primitives[2];
	for (unsigned int i=0;i<nrSamples;++i) {
	  float ettemp=0;

	  for (int ii=0;ii<nrFrames;++ii) {
	    int en=(mapEndcap_[thisTower][ii])[i].adc();
	    float et0 = TMath::Max(en- thresholds[ii],0);
	    et0=int(et0*eeDccAdcToGeV_/ebDccAdcToGeV_); 
	    float theta=theEndcapGeometry->getGeometry(mapEndcap_[thisTower][ii].id())->getPosition().theta();
	    et0 =(float) (et0*sin(theta));

	    ettemp += et0;
	  }
	  int et=int(ettemp);

	  //for the moment, there is no fgvb implemented...
	  int fgvb=0;

	  int ttf=calculateTTF(et);
	  et=et>>4;
	  if (et>0xFF) et=0xFF;
	  for (int nrt=0;nrt<nrTowers;++nrt) {
	    if (nrTowers==2)   primitives[nrt].push_back(EcalTriggerPrimitiveSample(et/2,fgvb,ttf));
	    //FIXME??
	    else primitives[nrt].push_back(EcalTriggerPrimitiveSample(et,fgvb,ttf));	  
	    
	  }
	}
	// Fill TriggerPrimitiveDigi
	for (int nrt=0;nrt<nrTowers;++nrt) {
	  tptow[nrt].setSize(nrSamplesToWrite_);
	  int isam=0;
	  for (int i=firstSample;i<=lastSample;++i) {
	    tptow[nrt].setSample(isam++,(primitives[nrt])[i]);
	  }
	  //	  LogDebug("EcalTPG") <<"For "<<thisTower<<" the following TP was created: "<<tptow[nrt];
	  result.push_back(tptow[nrt]);
	}
      } //end of loop over it
  }// !barrelOnly
}

//----------------------------------------------------------------------

void EcalTrigPrimFunctionalAlgo::fillBarrel(const EcalTrigTowerDetId & coarser, const EBDataFrame& samples) 
{
  // here we store a vector of EBDataFrames for each strip into a vector belonging to the corresponding tower
 
  int n=(((samples.id()).ic()-1)%100)/20; //20 corresponds to 4 * ecal_barrel_crystals_per_strip
  int stripnr;
  if ((samples.id()).ieta()<0) stripnr = n+1;
  else stripnr =ecal_barrel_strips_per_trigger_tower - n;
  SUMVB::iterator it= sumBarrel_.find(coarser);

  if(it==sumBarrel_.end()) 
    {
      for (int i=0;i<ecal_barrel_strips_per_trigger_tower;i++ ) {
 	std::vector<const EBDataFrame *>  truc;
	sumBarrel_[coarser].push_back(truc);
      } 
    }
  const EBDataFrame * p=& samples;
  (sumBarrel_[coarser])[stripnr-1].push_back(p);
    
}
//----------------------------------------------------------------------

void EcalTrigPrimFunctionalAlgo::fillEndcap(const EcalTrigTowerDetId & coarser, const EEDataFrame & frame){
  // temporary version, waiting for geometry of pseudostrips
  // for the moment we put into this map for each TT:
  // all the EEDataframes that belong to this tower
  //   SUMVE::iterator it = sumEndcap_.find(coarser);
  //   if(it==sumEndcap_.end()) {
  //     std::vector<int> sums(frame.MAXSAMPLES);
  //     it = (sumEndcap_.insert(SUMVE::value_type(coarser, sums))).first;
  //   }
  //   for (int i=0;i<frame.size();++i)  (*it).second[i] += frame[i].raw();

  MAPE::iterator it2 = mapEndcap_.find(coarser);
  if(it2==mapEndcap_.end()){ 
    std::vector<EEDataFrame> vec;
    it2 = (mapEndcap_.insert(MAPE::value_type(coarser, vec))).first;
  }
  (*it2).second.push_back(frame);

}

//----------------------------------------------------------------------
int EcalTrigPrimFunctionalAlgo::calculateTTF(const int en) {
  //temporary version of TTF calculation for Endcap
  //  int high=83; // adc value corresponding to 5 GeV
  //  int low=42;  // adc value corresponding to 2.5 GeV
  // temporary for temporary Endcap version !!!

  double threshLow_ =2.5; //GeV
  double threshHigh_=5.;
  //  
  int high=int(threshHigh_/ebDccAdcToGeV_);
  int low=int(threshLow_/ebDccAdcToGeV_);
  int ttf=0;
  if (en>high) ttf=3;
  else if (ttf<high && ttf >low ) ttf=2;
  return ttf;
}
//----------------------------------------------------------------------
 int EcalTrigPrimFunctionalAlgo::findTowerNrInSM(const EcalTrigTowerDetId &id) {
 // finds towr nr in supermodule in Barrel(from 1 to 68)
   const int nrphis=4;

   int ieta=id.ietaAbs();
   int iphi=id.iphi();
   int basenr=(ieta-1)*nrphis +1;
   int towernr=basenr+(iphi-1)%nrphis;
   return  towernr;
 }
