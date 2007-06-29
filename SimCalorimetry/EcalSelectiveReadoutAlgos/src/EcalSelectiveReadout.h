//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-"
/*
 * $Id: EcalSelectiveReadout.h,v 1.6 2007/02/14 18:37:04 pgras Exp $
 */

#ifndef ECALSELECTIVEREADOUT_H
#define ECALSELECTIVEREADOUT_H

#include <vector>
#include <iosfwd>
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#define ECALSELECTIVEREADOUT_NOGEOM //version w/o geometry dependency.

#ifndef ECALSELECTIVEREADOUT_NOGEOM
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#endif //ECALSELECTIVEREADOUT_NOGEOM not defined



/** This class is used to run the selective readout processing on the
 * electromagnetic calorimeter. Normal user do not need to access directly this
 * class: the selective readout is made at digitization by the
 * CaloDataFrameFormatter class which uses the EcalSelectiveReadout class for
 * this purpose: see CaloDataFrameFormatter class documentation.
 * The ECAL event size must be reduced by a factor ~1/20th online at the LV1
 * rate. This reduction is achieved by suppressing channels whose energy,
 * evaluated on line by a FIR, is below a threshold ("zero suppression"). This
 * threshold is dynamic and determined for the channel of each readout unit.
 * A readout unit is a group of 5x5 crystals: a trigger tower for the barrel
 * and a supercrystal for the endcap) basis. The selective readout classifies
 * the readout units in three classes:<UL>
 * <LI>low interest
 * <LI>single
 * <LI>neighbour
 * <LI>center
 * </UL>. Single, neighbour and center classes consitute the "high interest"
 * readout units and their channels the "high interest" channels.
 * Two zero suppresion thresholds, eventually -/+ infinity, are defined: one
 * for the "Low interest" channels and one for the "high interest" channels.
 * The low interest threshold must be higher than the high interest one.
 */
class EcalSelectiveReadout {
public:
  //type definitions
  /** trigger tower classification ("SRP flags")
   */
  typedef enum {UNKNOWN=-1,
                LOWINTEREST,
                SINGLE,
                NEIGHBOUR,
                CENTER,
                FORCED_RO} towerInterest_t;

  typedef enum {
    TTF_UNKNOWN=-1,
    TTF_LOW_INTEREST = 0x0,
    TTF_MID_INTEREST = 0x1,
    /* 0x2 not used */
    TTF_HIGH_INTEREST = 0X3,
    TTF_FORCED_RO_LINK_SYNC_ERR = 0x4,
    TTF_FORCED_RO_HAMMING_ERR = 0X5,
    TTF_FORCED_RO_OTHER1 = 0X6,
    TTF_FORCED_RO_OTHER2 = 0X7
  } ttFlag_t;

  static const int TTF_FORCED_RO_MASK = 0x4;
  
  //constants
public:
  /** Number of crystals along eta in barrel
   */
  const static size_t nBarrelEtaBins = 170;
  /** Number of crystals in a eta ring of the barrel
   */
  const static size_t nBarrelPhiBins = 360;
  /** Range of the x-index of endcap crystals (xman-xmin+1).
   */
  const static size_t nEndcapXBins = 100;
  /** Range of the y-index of endcap crystals (yman-ymin+1).
   */
  const static size_t nEndcapYBins = 100;
  /** Edge size of a supercrystal. A supercrystal is a tower of 5x5 crystals.
   */
  const static size_t supercrystalEdge = 5;
  /** Range of endcap supercrystal x-index (xmax-xmin+1)
   */
  const static size_t nSupercrystalXBins = nEndcapXBins/supercrystalEdge;
  /** Range of endcap supercrystal y-index (ymay-ymin+1)
   */
  const static size_t nSupercrystalYBins = nEndcapYBins/supercrystalEdge;
  /** Number of trigger tower along eta in the barrel
   */
  const static size_t nBarrelTowerEtaBins = nBarrelEtaBins/5;
  /** Number of trigger tower in a eta ring of the barrel
   */
  const static size_t nBarrelTowerPhiBins = nBarrelPhiBins/5;
  /** Number of endcap, obviously tow
   */
  const static size_t nEndcaps = 2;
  /** Number of trigger towers along eta in one endcap
   */
  const static size_t nEndcapTriggerTowersInEta = 11;
  /** Number of barrel trigger towers along eta
   */
  const static size_t nBarrelTriggerTowersInEta = 34;
  /** Number of trigger towers along eta for the whole ECAL
   */
  const static size_t nTriggerTowersInEta =
  2*nEndcapTriggerTowersInEta+nBarrelTriggerTowersInEta;
  /** Number of trigger towers in an eta ring
   */
  const static size_t nTriggerTowersInPhi = 72;
  
  //constructor(s) and destructor(s)
public:
  /** Constructs a ecalselectivereadout.
   * Neighbours are taken in a trigger tower matrix of size
   * 2(dEta+1))x2(dPhi+1) around a 'center' tower.
   * @param dEta neighborhood extend in number of trigger towers along eta
   * @param dPhi neighborgooh extend in number if trigger towers along phi
   * in 'low interest', 'single' or 'center. First element is the lower
   * threshold, second element is the higher one.
   */
  EcalSelectiveReadout(int dEta = 1, int dPhi = 1);

  //method(s)
  
  /// the mapping of which cell goes with which trigger tower
  void setTriggerMap(const EcalTrigTowerConstituentsMap * map) {
    theTriggerMap = map;
  }

#ifndef ECALSELECTIVEREADOUT_NOGEOM
  void setGeometry(const CaloGeometry * caloGeometry) {
    theGeometry = caloGeometry;
  }
#endif //ECALSELECTIVEREADOUT_NOGEOM not defined
  
  /** Selective readout algorithm type 0.
   *  The algorithm is the following:
   *  <OL>
   *  <LI>A trigger tower (TT) with Et higher than the high threshold is
   *  classified as 'center'
   *  <LI>A trigger tower which is a neighbour of a 'center' TT and which
   *  is not itself a 'center' is classified as 'neighbour'
   *  <LI>A TT with Et between the two threshold and which is not a 'center'
   *  or a 'neighbour' is classified as 'single'
   *  <LI>Any other TT are classified as 'low interest'
   *  </OL>
   * For the barrel a crystal inherit the
   *  single/center/neighbour/low_interst classification of its TT.
   *  For the endcap,
   *  <UL>
   *     <LI>if the supercrystal overlaps with a 'center' TT, it's flagged
   *  as 'center'
   *     <LI>otherwise if it overlaps with a 'neighbour' TT, it's flagged
   *  as 'neighbour'
   *     <LI>else if it overlaps with a 'single' TT, it's flagged
   *  as 'single'
   *     <LI>the remaining SC are flagged as 'low interest'
   *  </UL>
   * An endcap crystal inherits the SRP flag of its SC.
   * @param triggerTowerEt array of the transverse enrgy deposited in the
   * trigger tower. First index is for eta,2nd index for phi.
   */
  void runSelectiveReadout0(const ttFlag_t
                            towerFlags[nTriggerTowersInEta][nTriggerTowersInPhi]);

  /** Gets the SR interest classification of an EB channel
   * @param ebDetId id of the crystal
   * @param interest
   */
  towerInterest_t getCrystalInterest(const EBDetId & ebDetId) const;

  /** Gets the SR interest classification of an EE channel
   * @param eeDetId id of the crystal
   * @return interest
   */
  towerInterest_t getCrystalInterest(const EEDetId & eeDetId) const;

  /** Gets the SR interest classification of an EE supercrystal
   * @param scDetId id of the crystal
   * @return interest
   */
  towerInterest_t getSuperCrystalInterest(const EcalScDetId& scDetId) const;
  
  /**Gets the SR interest classification of a trigger tower (TT).
   * @param iEta index of the TT along eta
   * @param iPhi index of the TT along phi
   * @return interest
   */
  towerInterest_t getTowerInterest(const EcalTrigTowerDetId & towerId) const;

  /// print out header for the map: see print(std::ostream&)
  void printHeader(std::ostream & os) const;
  
  /// print out the map
  void print(std::ostream & os) const;

  void printBarrel(std::ostream & os) const;
  void printEndcap(int endcap, std::ostream & s) const;
  
private:

  /** Classifies trigger tower in three classes:<UL>
   * <LI> low interest: value 'lowInterest'
   * <LI> middle interest: value 'single'
   * <LI> high interest: value 'center'
   * </UL>
   */  
  void
  classifyTriggerTowers(const ttFlag_t ttFlags[nTriggerTowersInEta][nTriggerTowersInPhi]);
  
  
  /** Sets all supercrystal interest flags to 'unknown'
   */
  void resetSupercrystalInterest();
  
  /** Changes the value of a variable iff that has
   *  the effect to decrease the variable value
   *  var = min(var,val)
   *  @param var the variable
   *  @param val the new candidate value
   */
  void setLower(int& var, int val) const{
    if(val<var) var = val;
  }

  /** Changes the value of a variable iff that has
   *  the effect to increase the variable value
   *  var = max(var,val)
   *  @param var the variable
   *  @param val the new candidate value
   */
  template<class T>
  void setHigher(T& var, T val) const{
    if(val>var) var = val;
  }
  
  //attribute(s)
private:

  const EcalTrigTowerConstituentsMap * theTriggerMap;
#ifndef ECALSELECTIVEREADOUT_NOGEOM
  const CaloGeometry * theGeometry;
#endif //ECALSELECTIVEREADOUT_NOGEOM not defined
  towerInterest_t towerInterest[nTriggerTowersInEta][nTriggerTowersInPhi];
  towerInterest_t supercrystalInterest[nEndcaps][nSupercrystalXBins][nSupercrystalYBins];
  int dEta;
  int dPhi;

  // for printout
  const static char srpFlagMarker[];

};

std::ostream & operator<<(std::ostream & os, const EcalSelectiveReadout & selectiveReadout);

#endif
