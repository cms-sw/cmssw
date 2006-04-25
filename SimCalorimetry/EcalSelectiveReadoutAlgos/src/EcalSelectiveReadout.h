//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-"
/*
 * $Id$
 */

#ifndef ECALSELECTIVEREADOUT_H
#define ECALSELECTIVEREADOUT_H

#include <vector>
#include <iostream>

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
  typedef enum {unknown=-1,
                lowInterest,
                single,
                neighbour,
                center} towerInterest_t;
  
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
  
private:
  /** Name of the file containing endcap trigger tower definition, when
   * read from a file.
   */
  const static std::string fileName;
    
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
  
  /** Constructs a ecalselectivereadout.
   * Neighbours are taken in a trigger tower matrix of size
   * 2(dEta+1))x2(dPhi+1) around a 'center' tower.
   * @param thresholds thresholds for the trigger tower classification
   * in 'low interest', 'single' or 'center. First element is the lower
   * threshold, second element is the higher one.
   * @param dEta neighborhood extend in number of trigger towers along eta
   * @param dPhi neighborgooh extend in number if trigger towers along phi
   * @param towerMap crystal indices->trigger tower map
   */
  EcalSelectiveReadout(const std::vector<double>& thresholds,
                             const int* towerMap,
                             int dEta = 1, int dPhi = 1);

  //method(s)
public:
  /** Dumps trigger tower map.
   * @param out output stream to dump the tower map (default std::cout)
   */
  void dumpTriggerTowerMap(std::ostream& out=std::cout) const;
  
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
  std::vector<std::vector<towerInterest_t> >
  classifyTriggerTowers(const float towerEt[nTriggerTowersInEta][nTriggerTowersInPhi]) const;

  towerInterest_t getEndcapCrystalInterest(size_t iEndcap,
                                           size_t iX,
                                           size_t iY) const{    
    return endcapCrystalExists(iEndcap, iX, iY)?
      supercrystalInterest[iEndcap][supercrystalX(iX)][supercrystalY(iY)]:
      unknown;
  }
  
  
  towerInterest_t getBarrelCrystalInterest(size_t iEta, size_t iPhi) const{
    return towerInterest[barrelTowerEta(iEta)][barrelTowerPhi(iPhi)];
  }

  /**Gets the SR flags of a trigger tower (TT).
   * @param iEta index of the TT along eta
   * @param iPhi index of the TT along phi
   */
  towerInterest_t getTowerInterest(size_t iEta, size_t iPhi) const{
    return towerInterest[iEta][iPhi];
  }
  
private:

  /** Help function to get eta index of the TT of a given crystal.
   * @param iEta crystal 'natural' index
   */
  size_t barrelTowerEta(size_t iEta) const{
    return  nEndcapTriggerTowersInEta + (iEta/5);
  }

  /** Help function to get eta index of the TT of a given crystal.
   * @param iPhi crystal 'natural' index along phi
   */
  size_t barrelTowerPhi(size_t iPhi) const{
    return iPhi/5;
  }

  /** Help function to get x index of the SC of a given endcap crystal.
   * @param iX crystal 'natural' index along x axis starting at 0
   */
  size_t supercrystalX(size_t iX) const{
    return  iX/5;
  }

  /** Help function to get eta index of the SC of a given crystal.
   * @param iY crystal 'natural' index along y-axis starting at 0.
   */
  size_t supercrystalY(size_t iY) const{
    return iY/5;
  }
  
  /** For an incomplete supercrystal check if it contains the crystal
   * with a given index. Returns always true for a complete crystal. 
   * @param iEncap 0 for z<0 endcap, 1 for z>1 endcap
   * @param iX index of the crystal along x-axis starting at 0
   * @param iY index of the crystal along y-axis starting at 0
   */
  bool endcapCrystalExists(size_t iEndcap, size_t iX, size_t iY) const{
    //triggerTower array contains -1 for missing crystals and
    //positive value for existing ones. Let's exploit this property:
    return triggerTower[iEndcap][iX][iY][0] >= 0;
  }
  
  /** Sets all supercystal interest flags to 'unknown'
   */
  void resetSupercrystalInterest();
  
  std::vector<size_t> towerOfCrystal(size_t iEndcap, size_t iX,
                                     size_t iY) const;
  
  std::vector<std::vector<size_t> > crystalsOfTower(size_t iEta, size_t iPhi) const;
  
  void readEndcapTowerMap();

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
  std::vector<double> threshold;
  int triggerTower[nEndcaps][nEndcapXBins][nEndcapYBins][2];
  towerInterest_t towerInterest[nTriggerTowersInEta][nTriggerTowersInPhi];
  towerInterest_t supercrystalInterest[nEndcaps][nSupercrystalXBins][nSupercrystalYBins];
  int dEta;
  int dPhi;


};
#endif
