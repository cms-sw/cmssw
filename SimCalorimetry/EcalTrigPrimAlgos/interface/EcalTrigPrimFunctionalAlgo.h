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
  
  explicit EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup,int binofmax, bool tcpFormat, bool barrelOnly, bool debug, bool famos);

  virtual ~EcalTrigPrimFunctionalAlgo();

  /** this actually calculates the trigger primitives (from Digis) */

  void run(const edm::SortedCollection<EBDataFrame> * col, EcalTrigPrimDigiCollection & result, EcalTrigPrimDigiCollection & resultTcp);
  void run(const edm::SortedCollection<EEDataFrame> * col, EcalTrigPrimDigiCollection & result, EcalTrigPrimDigiCollection & resultTcp);
  void run_part1_EB(const edm::SortedCollection<EBDataFrame> * col);
  void run_part1_EE(const edm::SortedCollection<EEDataFrame> * col);
  template <class T> void run_part2(const edm::SortedCollection<T> * col, 
				    std::vector<std::vector<std::pair<int,std::vector<const T*> > > > &towerMap,
                                    EcalTrigPrimDigiCollection & result,
				    EcalTrigPrimDigiCollection & resultTcp);
  void updateESRecord(double ttfLowEB, double ttfHighEB, double ttfLowEE, double ttfHighEE);

 private:

  void init(const edm::EventSetup & setup);
  template <class T>  void initStructures(std::vector<std::vector<std::pair<int,std::vector<const T*> > > > & towMap);
  template <class T> void clean(std::vector<std::vector<std::pair<int,std::vector<const T*> > > > &towerMap);
  template <class T> void fillMap(const edm::SortedCollection<T> * col, std::vector<std::vector<std::pair<int,std::vector<const T*> > > > &towerMap);
  int findTccNr(const EcalTrigTowerDetId &id);
  int findTowerNrInTcc(const EcalTrigTowerDetId &id);
  int findStripNr(const EBDetId &id);
  int findStripNr(const EEDetId &id);

  // FIXME: temporary until hashedIndex works alsom for endcap
  int getIndex(const edm::SortedCollection<EBDataFrame> * col,EcalTrigTowerDetId& id) {return id.hashedIndex();}
  // mind that eta is continuous between barrel+endcap
  int getIndex(const edm::SortedCollection<EEDataFrame> * col,EcalTrigTowerDetId& id) {
    int ind=(id.ietaAbs()-18)*72 + id.iphi();
    if (id.zside()<0) ind+=792;
    return ind;
  }

  EcalFenixStrip * estrip_;
  EcalFenixTcp * etcp_;

  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;
  const CaloSubdetectorGeometry *theEndcapGeometry;
  const EcalElectronicsMapping* theMapping_;

  float threshold;

  int binOfMaximum_;
  int maxNrSamples_;

  bool tcpFormat_;
  bool barrelOnly_;
  bool debug_;  
  bool famos_;  

  const EcalTPParameters *ecaltpp_;

  static const unsigned int nrSamples_; //nr samples to write, should not be changed since by convention the size means that it is coming from simulation
  static const unsigned int maxNrSamplesOut_; //to be placed in the intermediate samples
  static const unsigned int maxNrTowers_; //FIXME: calculate from EcalTPParameters?
  static const unsigned int maxNrTPs_; //FIXME: calculate from EcalTPParameters?

  int nrTowers_;   // nr of towers found by fillmap method
  // data structures kept during the whole run
  std::vector<std::vector<int> > striptp_;
  std::vector<std::vector<std::pair<int,std::vector<const EBDataFrame *> > > > towerMapEB_;
  std::vector<std::vector<std::pair<int,std::vector<const EEDataFrame *> > > > towerMapEE_;
  std::vector<std::pair<int,EcalTrigTowerDetId> > hitTowers_;
  std::vector<EcalTriggerPrimitiveSample> towtp_;
  std::vector<EcalTriggerPrimitiveSample> towtp2_;
 };

//=================================== implementations =============================================

template <class T> void EcalTrigPrimFunctionalAlgo::run_part2(const edm::SortedCollection<T> * col, std::vector<std::vector<std::pair<int,std::vector<const T*> > > > &towerMap,
                                                        EcalTrigPrimDigiCollection & result,
							EcalTrigPrimDigiCollection & resultTcp)
{
  // prepare writing of TP-s

  int firstSample = binOfMaximum_-1 -nrSamples_/2;
  int lastSample = binOfMaximum_-1 +nrSamples_/2;
  int nrTP=0;

  for(int itow=0;itow<nrTowers_;itow++) 
    {

      int index=hitTowers_[itow].first;
      const EcalTrigTowerDetId thisTower=hitTowers_[itow].second;
      int towNr=findTowerNrInTcc(thisTower);
      int sectorNr= findTccNr(thisTower);

      // loop over all strips assigned to this trigger tower
      int nstr=0;
      for(unsigned int i = 0; i < towerMap[itow].size();++i)
	{
	  std::vector<const T *> df=(towerMap[index])[i].second;//vector of dataframes for this strip, size; nr of crystals/strip

	  if ((towerMap[index])[i].first > 0) {  
           estrip_->process(df,(towerMap[index])[i].first,i+1,towNr,sectorNr,striptp_[nstr++]);
	  }
	}//loop over strips in one tower

      std::vector<const T *> dummy;
      etcp_->process(dummy,striptp_,nstr,towtp_,towtp2_,sectorNr,towNr);

      // prepare TP-s
      // special treatment for 2 inner endcap rings
      int nrTowers;
      EcalTriggerPrimitiveDigi tptow[2];
      EcalTriggerPrimitiveDigi tptowTcp[2];
      if (thisTower.subDet()==EcalEndcap && (thisTower.ietaAbs()==27 || thisTower.ietaAbs()==28 ))
        {
          nrTowers=2;
          int phi=2*((thisTower.iphi()-1)/2);
          tptow[0]=EcalTriggerPrimitiveDigi(EcalTrigTowerDetId(thisTower.zside(),thisTower.subDet(),thisTower.ietaAbs(),phi+1));
          tptow[1]=EcalTriggerPrimitiveDigi(EcalTrigTowerDetId(thisTower.zside(),thisTower.subDet(),thisTower.ietaAbs(),phi+2));
	  if (tcpFormat_){
	    tptowTcp[0]=EcalTriggerPrimitiveDigi(EcalTrigTowerDetId(thisTower.zside(),thisTower.subDet(),thisTower.ietaAbs(),phi+1));
	    tptowTcp[1]=EcalTriggerPrimitiveDigi(EcalTrigTowerDetId(thisTower.zside(),thisTower.subDet(),thisTower.ietaAbs(),phi+2));
	  }
        }else {
          nrTowers=1;
          tptow[0]=EcalTriggerPrimitiveDigi(thisTower);
          if (tcpFormat_)   tptowTcp[0]=EcalTriggerPrimitiveDigi(thisTower);
	}

      // now fill in
      for (int nrt=0;nrt<nrTowers;nrt++) {
        (tptow[nrt]).setSize(nrSamples_);
        if (towtp_.size()<nrSamples_)  {  //FIXME: only once
          edm::LogWarning("") <<"Too few samples produced, nr is "<<towtp_.size();
          break;
        }
        int isam=0;
        for (int i=firstSample;i<=lastSample;++i) {
          if (nrTowers<=1)  tptow[nrt].setSample(isam++,EcalTriggerPrimitiveSample(towtp_[i]));
          else {
            float et=towtp_[i].compressedEt()/2.;
            tptow[nrt].setSample(isam++,EcalTriggerPrimitiveSample(et,towtp_[i].fineGrain(),towtp_[i].ttFlag()));
          }
        }
 	nrTP++;
        LogDebug("EcalTPG") <<" For tower "<<itow<<" created TP nr "<<nrTP<<" with Et "<<tptow[nrt].compressedEt();
        result.push_back(tptow[nrt]);
      }

      if (tcpFormat_) {

	for (int nrt=0;nrt<nrTowers;nrt++) {
	  tptowTcp[nrt].setSize(nrSamples_);
	  if (towtp2_.size()<nrSamples_)  {  //FIXME: only once
	    edm::LogWarning("") <<"Too few samples produced, nr is "<<towtp2_.size();
	    break;
	  }
	  int isam=0;
	  for (int i=firstSample;i<=lastSample;++i) {
	    if (nrTowers<=1)  tptowTcp[nrt].setSample(isam++,EcalTriggerPrimitiveSample(towtp2_[i]));
	    else {
	      float et=towtp2_[i].compressedEt()/2.;
	      tptowTcp[nrt].setSample(isam++,EcalTriggerPrimitiveSample(et,towtp2_[i].fineGrain(),towtp2_[i].ttFlag()));
	    }
	  }
	  resultTcp.push_back(tptowTcp[nrt]);
	}
      }
     } 
  return;
}

template <class T> void EcalTrigPrimFunctionalAlgo::fillMap(const edm::SortedCollection<T> * col, std::vector<std::vector<std::pair<int,std::vector<const T*> > > > &towerMap)
{
  // implementation for Barrel and Endcap

  if (col) {
    nrTowers_=0;
    LogDebug("EcalTPG") <<"Fill mapping, Collection size = "<<col->size();
    for(unsigned int i = 0; i < col->size() ; ++i) {
      const T &samples = (*col)[i]; 
      EcalTrigTowerDetId coarser=(*eTTmap_).towerOf(samples.id());
      int index=getIndex(col,coarser);
      int stripnr=findStripNr(samples.id());

      int filled=0;
      for (unsigned int ij=0;ij<towerMap[index].size();++ij) filled+=towerMap[index][ij].first;
      if (!filled) {
	hitTowers_[nrTowers_++]=std::pair <int,EcalTrigTowerDetId>(index,coarser);
      }

      //FIXME: temporary protection
      int ncryst=towerMap[index][stripnr-1].first;
       if (ncryst>=EcalTPParameters::nbMaxXtals_ ) {
        std::cout <<" !!!!!!!!!!!!! Too many xtals for TT "<<coarser<<" stripnr "<<stripnr<<" xtalid "<<samples.id()<<std::endl;
	continue;
      }
      ((towerMap[index])[stripnr-1].second)[ncryst]=&samples;
      (towerMap[index])[stripnr-1].first++;
    }
  
    LogDebug("EcalTPG")<<"fillMap"<<"[EcalTrigPrimFunctionalAlgo] (found " 
		       << col->size() << " frames in "<< towerMap.size() << " towers ";
  }
  else {
    LogDebug("EcalTPG")<<"FillMap - FillMap Collection size=0 !!!!";
  }
}

template <class T> void EcalTrigPrimFunctionalAlgo::clean( std::vector<std::vector<std::pair<int,std::vector<const T*> > > > & towMap) {  
  // clean internal data structures
  for (unsigned int i=0;i<maxNrTowers_;++i) 
      for (int j=0;j<EcalTPParameters::nbMaxStrips_ ;++j) (towMap[i])[j].first=0;
  
  return;
}
 
template <class T> void EcalTrigPrimFunctionalAlgo::initStructures( std::vector<std::vector<std::pair<int,std::vector<const T*> > > > & towMap) {  
  //initialise internal data structures
  std::vector <const T*> vec0(EcalTPParameters::nbMaxXtals_ );
  std::vector<std::pair<int,std::vector<const T *> > > vec1(EcalTPParameters::nbMaxStrips_);
  for (int i=0;i<EcalTPParameters::nbMaxStrips_ ;++i) vec1[i]=std::pair<int,std::vector<const T*> >(0,vec0);
  towMap.resize(maxNrTowers_); 
  for (unsigned int i=0;i<maxNrTowers_;++i) towMap[i]=vec1;
  
  std::vector<int> vecint(maxNrSamples_);
  striptp_.resize(EcalTPParameters::nbMaxStrips_);
  for (int i=0;i<EcalTPParameters::nbMaxStrips_;++i) striptp_[i]=vecint;
  
}

#endif
