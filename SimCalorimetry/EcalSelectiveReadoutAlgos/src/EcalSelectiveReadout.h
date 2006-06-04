//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-"
/*
 * $Id: EcalSelectiveReadout.h,v 1.3 2006/06/02 22:13:16 rpw Exp $
 */

#ifndef ECALSELECTIVEREADOUT_H
#define ECALSELECTIVEREADOUT_H

#include <vector>
#include <iosfwd>
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

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
                CENTER} towerInterest_t;
  
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
   * @param thresholds thresholds for the trigger tower classification
   * @param dEta neighborhood extend in number of trigger towers along eta
   * @param dPhi neighborgooh extend in number if trigger towers along phi
   * in 'low interest', 'single' or 'center. First element is the lower
   * threshold, second element is the higher one.
   */
  EcalSelectiveReadout(const std::vector<double>& thresholds,
                             int dEta = 1, int dPhi = 1);

  //method(s)
  
  /// the mapping of which cell goes with which trigger tower
  void setTriggerMap(const EcalTrigTowerConstituentsMap * map) {
    theTriggerMap = map;
  }


  void setGeometry(const CaloGeometry * caloGeometry) {
    theGeometry = caloGeometry;
  }

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
  void runSelectiveReadout0(const float
                            towerEt[nTriggerTowersInEta][nTriggerTowersInPhi]);
  
  /** Classifies trigger tower in three classes:<UL>
   * <LI> low interest: value 'lowInterest'
   * <LI> middle interest: value 'single'
   * <LI> high interest: value 'center'
   * </UL>
   */  
  void
  classifyTriggerTowers(const float towerEt[nTriggerTowersInEta][nTriggerTowersInPhi]);

  
  towerInterest_t getCrystalInterest(const EBDetId & ebDetId) const;

  towerInterest_t getCrystalInterest(const EEDetId & eeDetId) const;

  /**Gets the SR flags of a trigger tower (TT).
   * @param iEta index of the TT along eta
   * @param iPhi index of the TT along phi
   */
  towerInterest_t getTowerInterest(const EcalTrigTowerDetId & towerId) const;

  /// print out the map
  void print(std::ostream & os) const;

  void printBarrel(std::ostream & os) const;
  void printEndcap(int endcap, std::ostream & s) const;

private:

  
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
  const CaloGeometry * theGeometry;
  std::vector<double> threshold;
  towerInterest_t towerInterest[nTriggerTowersInEta][nTriggerTowersInPhi];
  towerInterest_t supercrystalInterest[nEndcaps][nSupercrystalXBins][nSupercrystalYBins];
  int dEta;
  int dPhi;

  // for printout
  const static char srpFlagMarker[];

};

std::ostream & operator<<(std::ostream & os, const EcalSelectiveReadout & selectiveReadout);

#endif
