#ifndef EcalEBPhase2TrigPrimAlgo_h
#define EcalEBPhase2TrigPrimAlgo_h
/** \class EcalEBPhase2TrigPrimAlgo
\author L. Lutton, N. Marinelli - Univ. of Notre Dame
  Description: forPhase II                                                       
  It uses the new Phase2 digis based onthe new EB electronics                                                                                                                                       
 This is the main algo which plugs in all the subcomponents for the
 amplitude and time measurement and the spike flagging
*/

#include <sys/time.h>
#include <iostream>
#include <vector>

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"

#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2Linearizer.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2AmplitudeReconstructor.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2TimeReconstructor.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2SpikeTagger.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2TPFormatter.h"

#include <map>
#include <utility>

class EcalTrigTowerDetId;
class ETPCoherenceTest;
class EcalEBTriggerPrimitiveSample;
class CaloSubdetectorGeometry;
class EBDataFrame_Ph2;

class EcalEBPhase2TrigPrimAlgo {
public:
  explicit EcalEBPhase2TrigPrimAlgo(const EcalTrigTowerConstituentsMap *eTTmap,
                                    const CaloGeometry *theGeometry,
                                    int binofmax,
                                    bool debug);

  virtual ~EcalEBPhase2TrigPrimAlgo();

  void run(const EBDigiCollectionPh2 *col, EcalEBTrigPrimDigiCollection &result);

  void setPointers(const EcalLiteDTUPedestalsMap *ecaltpPed,
                   const EcalEBPhase2TPGLinearizationConstMap *ecaltpLin,
                   const EcalTPGCrystalStatus *ecaltpgBadX,
                   const EcalEBPhase2TPGAmplWeightIdMap *ecaltpgAmplWeightMap,
                   const EcalEBPhase2TPGTimeWeightIdMap *ecaltpgTimeWeightMap,
                   const EcalTPGWeightGroup *ecaltpgWeightGroup) {
    ecaltpPed_ = ecaltpPed;
    ecaltpgBadX_ = ecaltpgBadX;
    ecaltpLin_ = ecaltpLin;
    ecaltpgAmplWeightMap_ = ecaltpgAmplWeightMap;
    ecaltpgTimeWeightMap_ = ecaltpgTimeWeightMap;
    ecaltpgWeightGroup_ = ecaltpgWeightGroup;
  }

private:
  //old void init(const edm::EventSetup & setup);
  void init();
  template <class T>
  void initStructures(std::vector<std::vector<std::pair<int, std::vector<T> > > > &towMap);
  template <class T>
  void clean(std::vector<std::vector<std::pair<int, std::vector<T> > > > &towerMap);

  void fillMap(EBDigiCollectionPh2 const *col,
               std::vector<std::vector<std::pair<int, std::vector<EBDataFrame_Ph2> > > > &towerMap);

  int findStripNr(const EBDetId &id);

  // FIXME: temporary until hashedIndex works alsom for endcap
  int getIndex(const EBDigiCollectionPh2 *, EcalTrigTowerDetId &id) { return id.hashedIndex(); }
  // mind that eta is continuous between barrel+endcap
  //  int getIndex(const  EEDigiCollectionPh2 *, EcalTrigTowerDetId& id) {
  // int ind=(id.ietaAbs()-18)*72 + id.iphi();
  // if (id.zside()<0) ind+=792;
  // return ind;
  // }

  const EcalTrigTowerConstituentsMap *eTTmap_ = nullptr;
  const CaloGeometry *theGeometry_ = nullptr;

  float threshold;
  int binOfMaximum_;
  int maxNrSamples_;

  bool barrelOnly_;
  bool debug_;

  int nrTowers_;  // nr of towers found by fillmap method
  static const unsigned int maxNrTowers_;
  static const unsigned int maxNrSamplesOut_;
  static const unsigned int nrSamples_;

  // data structures kept during the whole run
  std::vector<std::vector<int> > striptp_;
  std::vector<std::vector<std::pair<int, std::vector<EBDataFrame_Ph2> > > > towerMapEB_;
  std::vector<std::vector<std::pair<int, std::vector<EEDataFrame> > > > towerMapEE_;
  std::vector<std::pair<int, EcalTrigTowerDetId> > hitTowers_;
  std::vector<EcalEBTriggerPrimitiveSample> towtp_;
  std::vector<EcalEBTriggerPrimitiveSample> towtp2_;

  enum { nbMaxStrips_ = 5 };
  enum { nbMaxXtals_ = 5 };

  const EcalElectronicsMapping *theMapping_;

  EcalEBPhase2Linearizer *linearizer_;

  EcalEBPhase2AmplitudeReconstructor *amplitude_reconstructor_;
  EcalEBPhase2TimeReconstructor *time_reconstructor_;
  EcalEBPhase2SpikeTagger *spike_tagger_;
  EcalEBPhase2TPFormatter *tpFormatter_;

  //

  const EcalLiteDTUPedestalsMap *ecaltpPed_;
  const EcalTPGCrystalStatus *ecaltpgBadX_;

  const EcalTPGWeightGroup *ecaltpgWeightGroup_;
  const EcalEBPhase2TPGLinearizationConstMap *ecaltpLin_;
  const EcalEBPhase2TPGAmplWeightIdMap *ecaltpgAmplWeightMap_;
  const EcalEBPhase2TPGTimeWeightIdMap *ecaltpgTimeWeightMap_;

  EcalEBPhase2Linearizer *getLinearizer() const { return linearizer_; }
  std::vector<int> lin_out_;
  //
  EcalEBPhase2AmplitudeReconstructor *getAmplitudeFinder() const { return amplitude_reconstructor_; }
  std::vector<int> filt_out_;
  std::vector<int64_t> time_out_;
  std::vector<int> amp_out_;
  std::vector<int> outEt_;
  std::vector<int64_t> outTime_;

  EcalEBPhase2TimeReconstructor *getTimeFinder() const { return time_reconstructor_; }
  EcalEBPhase2SpikeTagger *getSpikeTagger() const { return spike_tagger_; }
  EcalEBPhase2TPFormatter *getTPFormatter() const { return tpFormatter_; }

  //
};

template <class T>
void EcalEBPhase2TrigPrimAlgo::clean(std::vector<std::vector<std::pair<int, std::vector<T> > > > &towMap) {
  // clean internal data structures
  for (unsigned int i = 0; i < maxNrTowers_; ++i)
    for (int j = 0; j < nbMaxStrips_; ++j)
      (towMap[i])[j].first = 0;
  return;
}

inline void EcalEBPhase2TrigPrimAlgo::fillMap(
    EBDigiCollectionPh2 const *col, std::vector<std::vector<std::pair<int, std::vector<EBDataFrame_Ph2> > > > &towerMap)

{
  // implementation for Barrel
  if (col) {
    nrTowers_ = 0;
    //if ( debug_) std::cout  <<"Fill mapping, Collection size = "<< col->size() << std::endl;;
    for (unsigned int i = 0; i < col->size(); ++i) {
      EBDigiCollectionPh2::Digi samples((*col)[i]);
      EcalTrigTowerDetId coarser = (*eTTmap_).towerOf(samples.id());
      int index = getIndex(col, coarser);
      EBDetId id = samples.id();
      int stripnr = findStripNr(id);

      int filled = 0;
      for (unsigned int ij = 0; ij < towerMap[index].size(); ++ij)
        filled += towerMap[index][ij].first;
      if (!filled) {
        hitTowers_[nrTowers_++] = std::pair<int, EcalTrigTowerDetId>(index, coarser);
      }

      //FIXME: temporary protection
      int ncryst = towerMap[index][stripnr - 1].first;
      if (ncryst >= nbMaxXtals_) {
        //edm::LogError("EcalTrigPrimFunctionAlgo")<<"! Too many xtals for TT "<<coarser<<" stripnr "<<stripnr<<" xtalid "<<samples.id() ;
        continue;
      }
      ((towerMap[index])[stripnr - 1].second)[ncryst] = samples;
      (towerMap[index])[stripnr - 1].first++;
    }

    if (debug_)
      std::cout << "fillMap"
                << "[EcalTrigPrimFunctionalAlgo] (found " << col->size() << " frames in " << towerMap.size()
                << " towers) " << std::endl;
  } else {
    if (debug_)
      std::cout << "FillMap - FillMap Collection size=0 !!!!" << std::endl;
    ;
  }
}

template <class T>
void EcalEBPhase2TrigPrimAlgo::initStructures(std::vector<std::vector<std::pair<int, std::vector<T> > > > &towMap) {
  //initialise internal data structures

  std::vector<T> vec0(nbMaxXtals_);
  std::vector<std::pair<int, std::vector<T> > > vec1(nbMaxStrips_);
  for (int i = 0; i < nbMaxStrips_; ++i)
    vec1[i] = std::pair<int, std::vector<T> >(0, vec0);
  towMap.resize(maxNrTowers_);
  for (unsigned int i = 0; i < maxNrTowers_; ++i)
    towMap[i] = vec1;

  std::vector<int> vecint(maxNrSamples_);
  striptp_.resize(nbMaxStrips_);
  for (int i = 0; i < nbMaxStrips_; ++i)
    striptp_[i] = vecint;
}

#endif
