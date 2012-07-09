#ifndef SIMCALORIMETRY_HCALSIMPRODUCERS_HCALHITRELABELLER_H
#define SIMCALORIMETRY_HCALSIMPRODUCERS_HCALHITRELABELLER_H 1


#include <vector>
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/** \class HcalHitRelabeller
  *  
  * $Date: 2011/03/12 17:11:23 $
  * $Revision: 1.1.2.2 $
  * \author J. Mans - Minnesota
  */
class HcalHitRelabeller {
public:
  HcalHitRelabeller(const edm::ParameterSet& ps);
  void process(const CrossingFrame<PCaloHit>& cf);
  CrossingFrame<PCaloHit> const* getCrossingFrame() { return m_crossFrame; }
  void clear();
private:
  DetId relabel(const uint32_t testId) const;

  std::vector<std::vector<int> > m_segmentation;

  CrossingFrame<PCaloHit>* m_crossFrame;
  std::vector<PCaloHit> m_signalRelabelled;
  // outer vector is for events, since crossingFrame needs everything
  // pinnned in memory
  std::vector<std::vector<PCaloHit> > m_pileupRelabelled;

};


#endif
