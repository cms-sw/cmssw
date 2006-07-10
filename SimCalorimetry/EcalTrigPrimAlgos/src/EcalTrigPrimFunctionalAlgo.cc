/** \class EcalTrigPrimFunctionalAlgo
 *
 * EcalTrigPrimFunctionalAlgo is the main algorithm class for TPG
 * It coordinates all the aother algorithms
 * Structi=ure is very close to electronics
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

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalTrigPrimFunctionalAlgo.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h"

#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include <TTree.h>

//----------------------------------------------------------------------

EcalTrigPrimFunctionalAlgo::EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup,int binofmax,int nrsamples, double lowthresh, double highthresh):valid_(false),valTree_(NULL),binOfMaximum_(binofmax),nrSamplesToWrite_(nrsamples), threshLow_(lowthresh), threshHigh_(highthresh)

{this->init(setup);}

//----------------------------------------------------------------------
EcalTrigPrimFunctionalAlgo::EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup,TTree *tree,int binofmax, int nrsamples, double lowthresh, double highthresh):valid_(true),valTree_(tree),binOfMaximum_(binofmax),nrSamplesToWrite_(nrsamples), threshLow_(lowthresh), threshHigh_(highthresh)
{this->init(setup);}

//----------------------------------------------------------------------
void EcalTrigPrimFunctionalAlgo::init(const edm::EventSetup & setup) {
  edm::ESHandle<CaloGeometry> theGeometry;
  //  edm::ESHandle<CaloSubdetectorGeometry> theBarrelGeometry;
  edm::ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle;
  setup.get<IdealGeometryRecord>().get( theGeometry );
  setup.get<IdealGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
  theEndcapGeometry = &(*theEndcapGeometry_handle);
  ebTopology_ = new EcalBarrelTopology(theGeometry);
  ebstrip_=new EcalBarrelFenixStrip(ebTopology_,valTree_);

  setup.get<IdealGeometryRecord>().get(eTTmap_);
}
//----------------------------------------------------------------------

EcalTrigPrimFunctionalAlgo::~EcalTrigPrimFunctionalAlgo() 
{
    delete ebstrip_;
    delete ebTopology_;
}

//----------------------------------------------------------------------

void EcalTrigPrimFunctionalAlgo::run(const EBDigiCollection* ebdcol,const EEDigiCollection* eedcol, EcalTrigPrimDigiCollection &  result, int fgvbMinEn) {
  
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
  for(unsigned int i = 0; i < ebdcol->size() ; ++i) {
    const EBDetId & myid=(*ebdcol)[i].id();
    const EcalTrigTowerDetId coarser= myid.tower();
    if(coarser.null())  
      {
	LogDebug("")<< "Cell " << myid << " has trivial coarser granularity (probably EFRY corner, not in this tower map; hit ignored)";
	continue;
      }	
	
   nhitsb++;
   fillBarrel(coarser,(*ebdcol)[i]);
  }// loop over all CaloDataFrames
  edm::LogInfo("")<< "[EcalTrigPrimFunctionalAlgo] (found " << nhitsb << " frames in " 
  	    << sumBarrel_.size() << " EBRY towers  ";

  
// loop over dataframes and fill map for endcap
   mapEndcap_.clear();
   for(unsigned int i = 0; i < eedcol->size() ; ++i) {
    const EEDetId & myid=(*eedcol)[i].id();
    EcalTrigTowerDetId coarser=(*eTTmap_).towerOf(myid);
    nhitse++;
    fillEndcap(coarser,(*eedcol)[i]);

  }// loop over all EEDataFrames
   edm::LogInfo("") << "[EcalTrigPrimFunctionalAlgo] (found " << nhitse << " frames in " 
  	    << mapEndcap_.size() << " EFRY towers  ";

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
	// loop over all strips assigned to this trigger tower
	std::vector<std::vector<int> > striptp;
        std::vector<std::vector<EBDataFrame> > dbgdf;
	for(unsigned int i = 0; i < min(it->second.size(),size_t(ecal_barrel_strips_per_trigger_tower)) ; ++i) 
           {
	    std::vector<int> tp;
 	    std::vector<EBDataFrame> df=it->second[i];

	    if (df.size()>0) {
	      tp=ebstrip_->process(df,i);
	      striptp.push_back(tp);

 	    }
	   }


	std::vector<EcalTriggerPrimitiveSample> towtp;
	ebtcp_.process(striptp,towtp);

	// Fill TriggerPrimitiveDigi
	EcalTriggerPrimitiveDigi tptow(thisTower);
	tptow.setSize(nrSamplesToWrite_);
        if (towtp.size()<nrSamplesToWrite_)  {  //UB FIXME: exception?
	  edm::LogWarning("Barrel") <<"Too few samples produced, nr is "<<towtp.size();
	  break;
	}
	int isam=0;
        for (int i=firstSample;i<=lastSample;++i) {
          tptow.setSample(isam++,EcalTriggerPrimitiveSample(towtp[i]));
	}
  	result.push_back(tptow);
      }

    //   Endcap treatment
    // completely temporary, waiting for endcap geometry !!!

    MAPE::const_iterator ite = mapEndcap_.begin(); 
    MAPE::const_iterator ee = mapEndcap_.end(); 

    itow=0;
    // loop over all trigger towers
    for(;ite!=ee;ite++) 
      {
        itow++;
        const EcalTrigTowerDetId & thisTower =(*ite).first;
	// loop over all strips assigned to this trigger tower
 	std::vector<int> striptp;

        int nrFrames=mapEndcap_[thisTower].size();
	// first, estimate thresholds
	LogDebug("treatEndcap")<<"\nStart TT "<<itow;
        std::vector<int>  thresholds(nrFrames);
	LogDebug("treatEndcap")<<"\nFor TT "<<itow<<", size of vector  "<<mapEndcap_[thisTower].size()<<" ID "<<thisTower;
        for (int ii=0;ii<nrFrames;++ii) {
	  thresholds[ii]=((mapEndcap_[thisTower][ii])[0].adc()+(mapEndcap_[thisTower][ii])[1].adc()+(mapEndcap_[thisTower][ii])[2].adc())/3;
	  LogDebug("treatEndcap")<<" Crystal "<< ii<<" threshold "<<thresholds[ii]<<", energies: ";
	  for (int j=0;j<(mapEndcap_[thisTower][ii]).size();++j) {
	    LogDebug("treatEndcap")<<" "<<(mapEndcap_[thisTower][ii])[j].adc();
	  }
	  LogDebug("treatEndcap")<<"\n";
	}

	int nrSamples=mapEndcap_[thisTower][0].size();
	std::vector<EcalTriggerPrimitiveSample> primitives;
	for (int i=0;i<nrSamples;++i) {
	  int et=0,etmax=0;

	  for (int ii=0;ii<nrFrames;++ii) {
	    int en=(mapEndcap_[thisTower][ii])[i].adc();
	    int et0 = TMath::Max(en- thresholds[ii],0);
	    et += et0;
            if (et0 > etmax) etmax=et0;
	  }

	  int fgvb=0;
	  if (etmax > fgvbMinEn && float(et)/float(etmax) > .85) fgvb=1;
	  LogDebug("EndcapTP")<<" For sample "<<i<<" summed et "<<et<<" fgvb "<<fgvb<<" etmax "<<etmax;
	  int ttf=calculateTTF(et);
          if (et>0xFF) et=0xFF;
	  primitives.push_back(EcalTriggerPrimitiveSample(et,fgvb,ttf));
	  LogDebug("EndcapTP")<<" For sample "<<i<<" summed et "<<et<<" fgvb "<<fgvb<<" etmax "<<etmax<<" ttf "<<ttf;
	}
	// Fill TriggerPrimitiveDigi
	EcalTriggerPrimitiveDigi tptow(thisTower);
	tptow.setSize(nrSamplesToWrite_);
        if (nrSamples<nrSamplesToWrite_)  { //UB FIXME: exception?
	  edm::LogWarning("Endcap") <<"Too few samples produced, nr is "<<nrSamples;
	  break;
	}
	int isam=0;
        for (int i=firstSample;i<=lastSample;++i) {
          tptow.setSample(isam++,primitives[i]);
	}
       	result.push_back(tptow);
      }
}


//----------------------------------------------------------------------

void EcalTrigPrimFunctionalAlgo::fillBarrel(const EcalTrigTowerDetId & coarser, const EBDataFrame& samples) 
{
  // here we store a vector of EBDataFrames for each strip into a vector belonging to the corresponding tower
 
  int stripnr = createStripNr( samples.id());
  SUMVB::iterator it= sumBarrel_.find(coarser);

  if(it==sumBarrel_.end()) 
    {
      for (int i=0;i<ecal_barrel_strips_per_trigger_tower;i++ ) {
 	std::vector<EBDataFrame>  truc;
	sumBarrel_[coarser].push_back(truc);
      } 
    }
  (sumBarrel_[coarser])[stripnr-1].push_back(samples);
    
}
//----------------------------------------------------------------------

void EcalTrigPrimFunctionalAlgo::fillEndcap(const EcalTrigTowerDetId & coarser, const EEDataFrame & frame){
  // temporary version, waiting for geometry of pseudostrips
  // for the moment we put into this map for each TT:
  //  the sum of the energies in all dataframes (vector<int>
  // the EEDataframes that belong to this tower
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
int EcalTrigPrimFunctionalAlgo::createStripNr(const EBDetId &cryst) {
  //  Calculates the stripnr in a barrel . Trigger Tower
  //  The stripnumber is found by counting how many times one can go to the east
  // before hitting the east border, starting from the given crystal

       int stripnr =ecal_barrel_strips_per_trigger_tower+1;
       EBDetId tmp=cryst;
       EcalBarrelNavigator nav(tmp,ebTopology_);
       uint32_t  towernumber = EcalTrigTowerDetId(tmp.tower()).rawId();
       while ( towernumber== EcalTrigTowerDetId(tmp.tower()).rawId()) {
  	--stripnr;
	tmp=nav.east();
        if (tmp.null()) {
          break; //no cell east of this one
	}
      }
      return  stripnr;
}


//----------------------------------------------------------------------
int EcalTrigPrimFunctionalAlgo::calculateTTF(const int en) {
  //temporary version of TTF calculation
  //  int high=83; // adc value corresponding to 5 GeV, factor 0.06
  //  int low=42;  // adc value corresponding to 2.5 GeV, factor 0.06
  int high=int(threshHigh_/0.06);
  int low=int(threshLow_/0.06);
  int ttf=0;
  if (en>high) ttf=3;
  else if (ttf<high && ttf >low ) ttf=2;
  return ttf;
}


