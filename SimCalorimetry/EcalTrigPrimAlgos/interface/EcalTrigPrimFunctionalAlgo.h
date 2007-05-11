#ifndef EcalTrigPrimFunctionalAlgo_h
#define EcalTrigPrimFunctionalAlgo_h
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
#include <sys/time.h>
#include <iostream>
#include <vector>

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStrip.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcp.h"

#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <map>
#include <utility>

/** Main Algo for Ecal trigger primitives. */


class TTree;
class EcalTrigTowerDetId;
class ETPCoherenceTest;
class EcalTriggerPrimitiveSample;
class CaloSubdetectorGeometry;
class EBDataFrame;
class EEDataFrame;
class EcalElectronicsMapping;

 
class EcalTrigPrimFunctionalAlgo
{  
 public:
  
 explicit EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup,int binofmax, int nrsamples, bool tcpFormat, bool barrelOnly, bool debug);
  explicit EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup, TTree *tree, int binofmax, int nrsamples, bool tcpFormat, bool barrelOnly,  bool debug);

  virtual ~EcalTrigPrimFunctionalAlgo();

  /** this actually calculates the trigger primitives (from Digis) */

   template <class T> void run(const edm::SortedCollection<T> * col, EcalTrigPrimDigiCollection & result, EcalTrigPrimDigiCollection & resultTcp);
void updateESRecord(double ttfLowEB, double ttfHighEB, double ttfLowEE, double ttfHighEE);

 private:

  void init(const edm::EventSetup & setup);

    template <class T> void fillMap(const edm::SortedCollection<T> * col,std::map<EcalTrigTowerDetId,std::vector<std::vector<const T*> >,std::less<EcalTrigTowerDetId> >  & towerMap,int & nhits); 
 
  int findTccNr(const EcalTrigTowerDetId &id);
  int findTowerNrInTcc(const EcalTrigTowerDetId &id);
  int findStripNr(const EBDetId &id);
  int findStripNr(const EEDetId &id);

  EcalFenixStrip * estrip_;
  EcalFenixTcp * etcp_;

  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;
  const CaloSubdetectorGeometry *theEndcapGeometry;
  const EcalElectronicsMapping* theMapping_;

  // for debugging
  ETPCoherenceTest *cTest_; //FIXME: remove

  //for validation
  bool valid_;
  TTree * valTree_;//FIXME: remove
 
  float threshold;

  int binOfMaximum_;
  unsigned int nrSamplesToWrite_;

  bool tcpFormat_;
  bool barrelOnly_;
  bool debug_;  

  const EcalTPParameters *ecaltpp_;

};

template <class T> void EcalTrigPrimFunctionalAlgo::run(const edm::SortedCollection<T> * col,
                                                        EcalTrigPrimDigiCollection & result,
							EcalTrigPrimDigiCollection & resultTcp)
{
  typedef std::map<EcalTrigTowerDetId,std::vector<std::vector<const T * > >,std::less<EcalTrigTowerDetId> > TOWMAP;
  TOWMAP towerMap;

  towerMap.clear();
 
  int nhits(0);
  
  // loop over dataframes and fill map 
  fillMap(col,towerMap,nhits);
  
  // prepare writing of TP-s

  int firstSample = binOfMaximum_-1 -nrSamplesToWrite_/2;
  int lastSample = binOfMaximum_-1 +nrSamplesToWrite_/2;
 
  typename TOWMAP::const_iterator it = towerMap.begin(); 
  typename TOWMAP::const_iterator e = towerMap.end(); 

  // loop over map and calculate TP-s
  int itow=0;
  // loop over all trigger towers
  for(;it!=e;it++) 
    {
      itow++;
      const EcalTrigTowerDetId & thisTower =(*it).first;

      int towNr=findTowerNrInTcc(thisTower);
      int sectorNr=findTccNr(thisTower);
      // loop over all strips assigned to this trigger tower
      std::vector<std::vector<int> > striptp;
      for(unsigned int i = 0; i < it->second.size() ; ++i)  
	{
	  std::vector<int> tp;
	  std::vector<const T *> df=it->second[i];

	  if (df.size()>0) {
	    tp=estrip_->process(df,i+1,towNr,sectorNr);
	    striptp.push_back(tp);
	  }
	}


      std::vector<EcalTriggerPrimitiveSample> towtp;
      std::vector<EcalTriggerPrimitiveSample> towtp2;
      std::vector<const T *> bid;
      etcp_->process(bid,striptp,towtp,towtp2,sectorNr,towNr); 

      // Fill TriggerPrimitiveDigi
      //      EcalTriggerPrimitiveDigi tptow(thisTower);
      //      tptow.setSize(nrSamplesToWrite_);
      //      if (towtp.size()<nrSamplesToWrite_)  { 
      //	edm::LogWarning("") <<"Too few samples produced, nr is "<<towtp.size();
      //	break;
      //      }
      // special treatment for 2 inner endcap rings
      int nrTowers;
      std::vector<EcalTriggerPrimitiveDigi> tptow;
      std::vector<EcalTriggerPrimitiveDigi> tptowTcp;
      if (thisTower.subDet()==EcalEndcap && (thisTower.ietaAbs()==27 || thisTower.ietaAbs()==28 ))
        {
          nrTowers=2;
          int phi=2*((thisTower.iphi()-1)/2);
          tptow.push_back(EcalTriggerPrimitiveDigi(EcalTrigTowerDetId(thisTower.zside(),thisTower.subDet(),thisTower.ietaAbs(),phi+1)));
          tptow.push_back(EcalTriggerPrimitiveDigi(EcalTrigTowerDetId(thisTower.zside(),thisTower.subDet(),thisTower.ietaAbs(),phi+2)));
	  if (tcpFormat_){
	    tptowTcp.push_back(EcalTriggerPrimitiveDigi(EcalTrigTowerDetId(thisTower.zside(),thisTower.subDet(),thisTower.ietaAbs(),phi+1)));
	    tptowTcp.push_back(EcalTriggerPrimitiveDigi(EcalTrigTowerDetId(thisTower.zside(),thisTower.subDet(),thisTower.ietaAbs(),phi+2)));
	  }
        }else {
          nrTowers=1;
          tptow.push_back(EcalTriggerPrimitiveDigi(thisTower));
          if (tcpFormat_)   tptowTcp.push_back(EcalTriggerPrimitiveDigi(thisTower));
	}

      /*       int isam=0; */
      /*       for (int i=firstSample;i<=lastSample;++i) { */
      /* 	tptow.setSample(isam++,EcalTriggerPrimitiveSample(towtp[i])); */
      /*       } */
      /*       result.push_back(tptow); */

      for (int nrt=0;nrt<nrTowers;nrt++) {
        tptow[nrt].setSize(nrSamplesToWrite_);
        if (towtp.size()<nrSamplesToWrite_)  {  //FIXME: only once
          edm::LogWarning("") <<"Too few samples produced, nr is "<<towtp.size();
          break;
        }
        int isam=0;
        for (int i=firstSample;i<=lastSample;++i) {
          if (nrTowers<=1)  tptow[nrt].setSample(isam++,EcalTriggerPrimitiveSample(towtp[i]));
          else {
            float et=towtp[i].compressedEt()/2.;
            tptow[nrt].setSample(isam++,EcalTriggerPrimitiveSample(et,towtp[i].fineGrain(),towtp[i].ttFlag()));
          }
        }
        result.push_back(tptow[nrt]);
      }

      if (tcpFormat_) {
	for (int nrt=0;nrt<nrTowers;nrt++) {
	  tptowTcp[nrt].setSize(nrSamplesToWrite_);
	  if (towtp2.size()<nrSamplesToWrite_)  {  //FIXME: only once
	    edm::LogWarning("") <<"Too few samples produced, nr is "<<towtp2.size();
	    break;
	  }
	  int isam=0;
	  for (int i=firstSample;i<=lastSample;++i) {
	    if (nrTowers<=1)  tptowTcp[nrt].setSample(isam++,EcalTriggerPrimitiveSample(towtp2[i]));
	    else {
	      float et=towtp2[i].compressedEt()/2.;
	      tptowTcp[nrt].setSample(isam++,EcalTriggerPrimitiveSample(et,towtp2[i].fineGrain(),towtp2[i].ttFlag()));
	    }
	  }
	  resultTcp.push_back(tptowTcp[nrt]);
	}
      }
      /* 	EcalTriggerPrimitiveDigi tptow(thisTower); */
      /* 	tptow.setSize(nrSamplesToWrite_); */
      /* 	if (towtp2.size()<nrSamplesToWrite_)  {  */
      /* 	  edm::LogWarning("") <<"Too few samples produced, nr is "<<towtp.size(); */
      /* 	  break; */
      /* 	} */
      /* 	int isam=0; */
      /* 	for (int i=firstSample;i<=lastSample;++i) { */
      /* 	  tptow.setSample(isam++,EcalTriggerPrimitiveSample(towtp2[i])); */
      /* 	} */
      /* 	resultTcp.push_back(tptow); */
      /*       } */
    }
}
 
template <class T> void EcalTrigPrimFunctionalAlgo::fillMap(const edm::SortedCollection<T> * col,std::map<EcalTrigTowerDetId,std::vector<std::vector<const T*> >,std::less<EcalTrigTowerDetId> >  & towerMap,int & nhits)
{
  // implementation for Barrel and Endcap

  typedef std::map<EcalTrigTowerDetId,std::vector<std::vector<const T * > >,std::less<EcalTrigTowerDetId> > TOWMAP;
  
  if (col) {
    LogDebug("EcalTPG") <<"Fill mapping, Collection size = "<<col->size();
    for(unsigned int i = 0; i < col->size() ; ++i) {
      const T &samples = (*col)[i];
      EcalTrigTowerDetId coarser=(*eTTmap_).towerOf(samples.id());
      int stripnr=findStripNr(samples.id());
      nhits++;
      typename TOWMAP::const_iterator it= towerMap.find(coarser);
      if(it==towerMap.end()) {
	//        for (int i=0;i<ecal_endcap_max_strips_per_trigger_tower;i++ ) {
       for (int i=0;i<EcalTPParameters::nbMaxStrips_;i++ ) {
          std::vector<const T *>  vec;
          towerMap[coarser].push_back(vec);
        } 
      }
      const T * p=& samples;
      //FIXME: temporary protection
      if ((towerMap[coarser])[stripnr-1].size()<EcalTPParameters::nbMaxXtals_ ) {
	(towerMap[coarser])[stripnr-1].push_back(p);
      }else if (debug_) {
	std::cout <<" !!!!!!!!!!!!! Too many xtals for TT "<<coarser<<" stripnr "<<stripnr<<" xtalid "<<p->id()<<std::endl;
	for (unsigned int kk=0;kk<(towerMap[coarser])[stripnr-1].size();kk++)
	  std::cout<<"xtal "<<kk<<" detid "<<((towerMap[coarser])[stripnr-1])[kk]->id()<<" towerid "<<(*eTTmap_).towerOf(((towerMap[coarser])[stripnr-1])[kk]->id())<<std::endl;
      }
    
    }
    LogDebug("EcalTPG")<<"fillMap"<<"[EcalTrigPrimFunctionalAlgo] (found " 
		       << nhits << " frames in "<< towerMap.size() << " towers ";
  }
  else {
    LogDebug("EcalTPG")<<"FillMap - FillMap Collection size=0 !!!!";
  }
}

#endif
