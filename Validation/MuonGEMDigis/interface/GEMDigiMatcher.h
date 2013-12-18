#ifndef GEMValidation_GEMDigiMatcher_h
#define GEMValidation_GEMDigiMatcher_h

/**\class DigiMatcher

 Description: Matching of Digis for SimTrack in GEM

 Original Author:  "Vadim Khotilovich"
 $Id: GEMDigiMatcher.h,v 1.1 2013/02/11 07:33:07 khotilov Exp $
*/

#include "DigiMatcher.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"

#include <vector>
#include <map>
#include <set>

class SimHitMatcher;

class GEMDigiMatcher : public DigiMatcher
{
public:

  GEMDigiMatcher(SimHitMatcher& sh);
  
  ~GEMDigiMatcher();

  // partition GEM detIds with digis
  std::set<unsigned int> detIds() const;

  // chamber detIds with digis
  std::set<unsigned int> chamberIds() const;

  // superchamber detIds with digis
  std::set<unsigned int> superChamberIds() const;

  // partition detIds with coincidence pads
  std::set<unsigned int> detIdsWithCoPads() const;

  // superchamber detIds with coincidence pads
  std::set<unsigned int> superChamberIdsWithCoPads() const;


  // GEM digis from a particular partition, chamber or superchamber
  const DigiContainer& digisInDetId(unsigned int) const;
  const DigiContainer& digisInChamber(unsigned int) const;
  const DigiContainer& digisInSuperChamber(unsigned int) const;

  // GEM pads from a particular partition, chamber or superchamber
  const DigiContainer& padsInDetId(unsigned int) const;
  const DigiContainer& padsInChamber(unsigned int) const;
  const DigiContainer& padsInSuperChamber(unsigned int) const;

  // GEM co-pads from a particular partition or superchamber
  const DigiContainer& coPadsInDetId(unsigned int) const;
  const DigiContainer& coPadsInSuperChamber(unsigned int) const;

  // #layers with digis from this simtrack
  int nLayersWithDigisInSuperChamber(unsigned int) const;
  int nLayersWithPadsInSuperChamber(unsigned int) const;

  /// How many pads in GEM did this simtrack get in total?
  int nPads() const;

  /// How many coincidence pads in GEM did this simtrack get in total?
  int nCoPads() const;

  std::set<int> stripNumbersInDetId(unsigned int) const;
  std::set<int> padNumbersInDetId(unsigned int) const;
  std::set<int> coPadNumbersInDetId(unsigned int) const;

  // what unique partitions numbers with digis from this simtrack?
  std::set<int> partitionNumbers() const;
  std::set<int> partitionNumbersWithCoPads() const;

private:

  void init();

  void matchDigisToSimTrack(const GEMDigiCollection& digis);
  void matchPadsToSimTrack(const GEMCSCPadDigiCollection& pads);
  void matchCoPadsToSimTrack(const GEMCSCPadDigiCollection& co_pads);

  edm::InputTag gemDigiInput_;
  edm::InputTag gemPadDigiInput_;
  edm::InputTag gemCoPadDigiInput_;

  int minBXGEM_, maxBXGEM_;

  int matchDeltaStrip_;

  std::map<unsigned int, DigiContainer> detid_to_digis_;
  std::map<unsigned int, DigiContainer> chamber_to_digis_;
  std::map<unsigned int, DigiContainer> superchamber_to_digis_;

  std::map<unsigned int, DigiContainer> detid_to_pads_;
  std::map<unsigned int, DigiContainer> chamber_to_pads_;
  std::map<unsigned int, DigiContainer> superchamber_to_pads_;

  std::map<unsigned int, DigiContainer> detid_to_copads_;
  std::map<unsigned int, DigiContainer> chamber_to_copads_;
  std::map<unsigned int, DigiContainer> superchamber_to_copads_;
};

#endif
