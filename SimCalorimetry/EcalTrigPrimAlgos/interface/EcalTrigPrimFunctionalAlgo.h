#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALTRIGPRIMFUNCTIONALALGO_h
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALTRIGPRIMFUNCTIONALALGO_h
/** \class EcalTrigPrimFunctionalAlgo
 *
 * EcalTrigPrimFunctionalAlgo is the main algorithm class for TPG
 * It coordinates all the aother algorithms
 * Structure is as close as possible to electronics
 *
 *
 * \author Ursula Berthon, Stephanie Baffioni, LLR Palaiseau
 *
 * \version   1st Version may 2006
 * \version   2nd Version jul 2006
 * \version   3rd Version sep 2007  introducing new Records closer to the db

 *
 ************************************************************/
#include <iostream>
#include <sys/time.h>
#include <vector>

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStrip.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcp.h"

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>
#include <utility>
#include <string>

/** Main Algo for Ecal trigger primitives. */

class EcalTrigTowerDetId;
class ETPCoherenceTest;
class EcalTriggerPrimitiveSample;
class CaloSubdetectorGeometry;
class EBDataFrame;
class EEDataFrame;
class EcalElectronicsMapping;

class EcalTrigPrimFunctionalAlgo {
public:
  //Not barrelOnly
  EcalTrigPrimFunctionalAlgo(const EcalTrigTowerConstituentsMap *eTTmap,
                             const CaloSubdetectorGeometry *endcapGeometry,
                             const EcalElectronicsMapping *theMapping,
                             int binofmax,
                             bool tcpFormat,
                             bool debug,
                             bool famos,
                             bool TPinfoPrintout);

  //barrel only
  explicit EcalTrigPrimFunctionalAlgo(const EcalElectronicsMapping *theMapping,
                                      int binofmax,
                                      bool tcpFormat,
                                      bool debug,
                                      bool famos,
                                      bool TPinfoPrintout);

  virtual ~EcalTrigPrimFunctionalAlgo();

  void run(const EBDigiCollection *col, EcalTrigPrimDigiCollection &result, EcalTrigPrimDigiCollection &resultTcp);
  void run(const EEDigiCollection *col, EcalTrigPrimDigiCollection &result, EcalTrigPrimDigiCollection &resultTcp);
  void run_part1_EB(EBDigiCollection const *col);
  void run_part1_EE(EEDigiCollection const *col);
  template <class Coll>
  void run_part2(Coll const *col,
                 std::vector<std::vector<std::pair<int, std::vector<typename Coll::Digi>>>> &towerMap,
                 EcalTrigPrimDigiCollection &result,
                 EcalTrigPrimDigiCollection &resultTcp);

  void setPointers(const EcalTPGLinearizationConst *ecaltpLin,
                   const EcalTPGPedestals *ecaltpPed,
                   const EcalTPGSlidingWindow *ecaltpgSlidW,
                   const EcalTPGWeightIdMap *ecaltpgWeightMap,
                   const EcalTPGWeightGroup *ecaltpgWeightGroup,
                   const EcalTPGOddWeightIdMap *ecaltpgOddWeightMap,
                   const EcalTPGOddWeightGroup *ecaltpgOddWeightGroup,
                   const EcalTPGFineGrainStripEE *ecaltpgFgStripEE,
                   const EcalTPGCrystalStatus *ecaltpgBadX,
                   const EcalTPGStripStatus *ecaltpgStripStatus,
                   const EcalTPGTPMode *ecaltpgTPMode) {
    estrip_->setPointers(ecaltpPed,
                         ecaltpLin,
                         ecaltpgWeightMap,
                         ecaltpgWeightGroup,
                         ecaltpgOddWeightMap,
                         ecaltpgOddWeightGroup,
                         ecaltpgSlidW,
                         ecaltpgFgStripEE,
                         ecaltpgBadX,
                         ecaltpgStripStatus,
                         ecaltpgTPMode);
  }
  void setPointers2(const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,
                    const EcalTPGLutGroup *ecaltpgLutGroup,
                    const EcalTPGLutIdMap *ecaltpgLut,
                    const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB,
                    const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE,
                    const EcalTPGTowerStatus *ecaltpgBadTT,
                    const EcalTPGSpike *ecaltpgSpike,
                    const EcalTPGTPMode *ecaltpgTPMode) {
    etcp_->setPointers(ecaltpgFgEBGroup,
                       ecaltpgLutGroup,
                       ecaltpgLut,
                       ecaltpgFineGrainEB,
                       ecaltpgFineGrainTowerEE,
                       ecaltpgBadTT,
                       ecaltpgSpike,
                       ecaltpgTPMode);
  }

private:
  void init();
  template <class T>
  void initStructures(std::vector<std::vector<std::pair<int, std::vector<T>>>> &towMap);
  template <class T>
  void clean(std::vector<std::vector<std::pair<int, std::vector<T>>>> &towerMap);
  template <class Coll>
  void fillMap(Coll const *col, std::vector<std::vector<std::pair<int, std::vector<typename Coll::Digi>>>> &towerMap);
  int findStripNr(const EBDetId &id);
  int findStripNr(const EEDetId &id);

  // FIXME: temporary until hashedIndex works alsom for endcap
  int getIndex(const EBDigiCollection *, EcalTrigTowerDetId &id) { return id.hashedIndex(); }
  // mind that eta is continuous between barrel+endcap
  int getIndex(const EEDigiCollection *, EcalTrigTowerDetId &id) {
    int ind = (id.ietaAbs() - 18) * 72 + id.iphi();
    if (id.zside() < 0)
      ind += 792;
    return ind;
  }

  std::unique_ptr<EcalFenixStrip> estrip_;
  std::unique_ptr<EcalFenixTcp> etcp_;

  const EcalTrigTowerConstituentsMap *eTTmap_ = nullptr;
  const CaloSubdetectorGeometry *theEndcapGeometry_ = nullptr;
  const EcalElectronicsMapping *theMapping_;

  float threshold;

  int binOfMaximum_;
  int maxNrSamples_;

  bool tcpFormat_;
  bool barrelOnly_;
  bool debug_;
  bool famos_;
  bool tpInfoPrintout_;

  static const unsigned int nrSamples_;        // nr samples to write, should not be changed since by
                                               // convention the size means that it is coming from simulation
  static const unsigned int maxNrSamplesOut_;  // to be placed in the intermediate samples
  static const unsigned int maxNrTowers_;      // would be better to get from somewhere..
  static const unsigned int maxNrTPs_;         // would be better to get from
                                               // somewhere..

  int nrTowers_;  // nr of towers found by fillmap method

  // data structures kept during the whole run
  std::vector<std::vector<int>> striptp_;
  std::vector<std::vector<std::pair<int, std::vector<EBDataFrame>>>> towerMapEB_;
  std::vector<std::vector<std::pair<int, std::vector<EEDataFrame>>>> towerMapEE_;
  std::vector<std::pair<int, EcalTrigTowerDetId>> hitTowers_;
  std::vector<EcalTriggerPrimitiveSample> towtp_;
  std::vector<EcalTriggerPrimitiveSample> towtp2_;

  enum { nbMaxStrips_ = 5 };
  enum { nbMaxXtals_ = 5 };
};

//=================================== implementations
//=============================================

template <class Coll>
void EcalTrigPrimFunctionalAlgo::run_part2(
    Coll const *col,
    std::vector<std::vector<std::pair<int, std::vector<typename Coll::Digi>>>> &towerMap,
    EcalTrigPrimDigiCollection &result,
    EcalTrigPrimDigiCollection &resultTcp) {
  typedef typename Coll::Digi Digi;

  // prepare writing of TP-s

  int firstSample = binOfMaximum_ - 1 - nrSamples_ / 2;
  int lastSample = binOfMaximum_ - 1 + nrSamples_ / 2;
  int nrTP = 0;
  std::vector<typename Coll::Digi> dummy;
  EcalTriggerPrimitiveDigi tptow[2];
  EcalTriggerPrimitiveDigi tptowTcp[2];

  estrip_->getFGVB()->setbadStripMissing(false);

  for (int itow = 0; itow < nrTowers_; ++itow) {
    int index = hitTowers_[itow].first;
    const EcalTrigTowerDetId &thisTower = hitTowers_[itow].second;

    if (tpInfoPrintout_) {
      edm::LogVerbatim("EcalTPG") << "+++++++++++++++++++++++++++++++++++++++++++++++++";
      edm::LogVerbatim("EcalTPG") << "on Tower " << itow << " of " << nrTowers_;
      edm::LogVerbatim("EcalTPG") << "Tower ieta, iphi: " << thisTower.ieta() << ", " << thisTower.iphi();
    }

    // loop over all strips assigned to this trigger tower
    int nstr = 0;
    for (unsigned int i = 0; i < towerMap[itow].size(); ++i) {
      std::vector<Digi> &df = (towerMap[index])[i].second;  // vector of dataframes for this strip,
                                                            // size; nr of crystals/strip

      if ((towerMap[index])[i].first > 0) {
        if (tpInfoPrintout_) {
          edm::LogVerbatim("EcalTPG") << "-------------------------------------------------";
          edm::LogVerbatim("EcalTPG") << "on Strip index " << i;
        }
        estrip_->process(df, (towerMap[index])[i].first, striptp_[nstr++]);
      }
    }  // loop over strips in one tower

    bool isInInnerRings = false;
    if (thisTower.subDet() == EcalEndcap && (thisTower.ietaAbs() == 27 || thisTower.ietaAbs() == 28))
      isInInnerRings = true;
    etcp_->process(dummy, striptp_, nstr, towtp_, towtp2_, isInInnerRings, thisTower);

    // prepare TP-s
    // special treatment for 2 inner endcap rings
    int nrTowers;
    if (isInInnerRings) {
      nrTowers = 2;
      int phi = 2 * ((thisTower.iphi() - 1) / 2);
      tptow[0] = EcalTriggerPrimitiveDigi(
          EcalTrigTowerDetId(thisTower.zside(), thisTower.subDet(), thisTower.ietaAbs(), phi + 1));
      tptow[1] = EcalTriggerPrimitiveDigi(
          EcalTrigTowerDetId(thisTower.zside(), thisTower.subDet(), thisTower.ietaAbs(), phi + 2));

      if (tcpFormat_) {
        tptowTcp[0] = EcalTriggerPrimitiveDigi(
            EcalTrigTowerDetId(thisTower.zside(), thisTower.subDet(), thisTower.ietaAbs(), phi + 1));
        tptowTcp[1] = EcalTriggerPrimitiveDigi(
            EcalTrigTowerDetId(thisTower.zside(), thisTower.subDet(), thisTower.ietaAbs(), phi + 2));
      }
    } else {
      nrTowers = 1;
      tptow[0] = EcalTriggerPrimitiveDigi(thisTower);
      if (tcpFormat_)
        tptowTcp[0] = EcalTriggerPrimitiveDigi(thisTower);
    }

    // now fill in
    for (int nrt = 0; nrt < nrTowers; nrt++) {
      (tptow[nrt]).setSize(nrSamples_);
      if (towtp_.size() < nrSamples_) {  // FIXME: only once
        edm::LogWarning("") << "Too few samples produced, nr is " << towtp_.size();
        break;
      }
      int isam = 0;
      for (int i = firstSample; i <= lastSample; ++i) {
        tptow[nrt].setSample(isam++, EcalTriggerPrimitiveSample(towtp_[i]));
      }
      nrTP++;
      LogDebug("EcalTPG") << " For tower " << itow << " created TP nr " << nrTP << " with Et "
                          << tptow[nrt].compressedEt();
      result.push_back(tptow[nrt]);
    }

    if (tcpFormat_) {
      for (int nrt = 0; nrt < nrTowers; nrt++) {
        tptowTcp[nrt].setSize(nrSamples_);
        if (towtp2_.size() < nrSamples_) {  // FIXME: only once
          edm::LogWarning("") << "Too few samples produced, nr is " << towtp2_.size();
          break;
        }
        int isam = 0;
        for (int i = firstSample; i <= lastSample; ++i) {
          if (nrTowers <= 1)
            tptowTcp[nrt].setSample(isam++, EcalTriggerPrimitiveSample(towtp2_[i]));
          else {
            int et = towtp2_[i].compressedEt() / 2;
            tptowTcp[nrt].setSample(isam++,
                                    EcalTriggerPrimitiveSample(et, towtp2_[i].fineGrain(), towtp2_[i].ttFlag()));
          }
        }
        resultTcp.push_back(tptowTcp[nrt]);
      }
    }
  }
  return;
}

template <class Coll>
void EcalTrigPrimFunctionalAlgo::fillMap(
    Coll const *col, std::vector<std::vector<std::pair<int, std::vector<typename Coll::Digi>>>> &towerMap) {
  typedef typename Coll::Digi Digi;

  // implementation for Barrel and Endcap

  if (col) {
    nrTowers_ = 0;
    LogDebug("EcalTPG") << "Fill mapping, Collection size = " << col->size();
    for (unsigned int i = 0; i < col->size(); ++i) {
      Digi samples((*col)[i]);
      EcalTrigTowerDetId coarser =
          eTTmap_ ? (*eTTmap_).towerOf(samples.id()) : EcalTrigTowerConstituentsMap::barrelTowerOf(samples.id());
      int index = getIndex(col, coarser);
      int stripnr = findStripNr(samples.id());

      int filled = 0;
      for (unsigned int ij = 0; ij < towerMap[index].size(); ++ij)
        filled += towerMap[index][ij].first;
      if (!filled) {
        hitTowers_[nrTowers_++] = std::pair<int, EcalTrigTowerDetId>(index, coarser);
      }

      // FIXME: temporary protection
      int ncryst = towerMap[index][stripnr - 1].first;
      if (ncryst >= nbMaxXtals_) {
        edm::LogError("EcalTrigPrimFunctionAlgo")
            << "! Too many xtals for TT " << coarser << " stripnr " << stripnr << " xtalid " << samples.id();
        continue;
      }
      ((towerMap[index])[stripnr - 1].second)[ncryst] = samples;
      (towerMap[index])[stripnr - 1].first++;
    }

    LogDebug("EcalTPG") << "fillMap"
                        << "[EcalTrigPrimFunctionalAlgo] (found " << col->size() << " frames in " << towerMap.size()
                        << " towers) ";
  } else {
    LogDebug("EcalTPG") << "FillMap - FillMap Collection size=0 !!!!";
  }
}

template <class T>
void EcalTrigPrimFunctionalAlgo::clean(std::vector<std::vector<std::pair<int, std::vector<T>>>> &towMap) {
  // clean internal data structures
  for (unsigned int i = 0; i < maxNrTowers_; ++i)
    for (int j = 0; j < nbMaxStrips_; ++j)
      (towMap[i])[j].first = 0;
  return;
}

template <class T>
void EcalTrigPrimFunctionalAlgo::initStructures(std::vector<std::vector<std::pair<int, std::vector<T>>>> &towMap) {
  // initialise internal data structures

  std::vector<T> vec0(nbMaxXtals_);
  std::vector<std::pair<int, std::vector<T>>> vec1(nbMaxStrips_);
  for (int i = 0; i < nbMaxStrips_; ++i)
    vec1[i] = std::pair<int, std::vector<T>>(0, vec0);
  towMap.resize(maxNrTowers_);
  for (unsigned int i = 0; i < maxNrTowers_; ++i)
    towMap[i] = vec1;

  std::vector<int> vecint(maxNrSamples_);
  striptp_.resize(nbMaxStrips_);
  for (int i = 0; i < nbMaxStrips_; ++i)
    striptp_[i] = vecint;
}

#endif
