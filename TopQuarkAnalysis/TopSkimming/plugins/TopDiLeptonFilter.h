#ifndef TopSkimming_TopDiLeptonFilter_h
#define TopSkimming_TopDiLeptonFilter_h
// Original Author:  Dmytro Kovalskyi, UCSB
// $Id$
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
class TopDiLeptonFilter : public edm::EDFilter {
 public:
   explicit TopDiLeptonFilter(const edm::ParameterSet&);
   virtual ~TopDiLeptonFilter() {}
 private:
   virtual bool filter(edm::Event&, const edm::EventSetup&);
   edm::InputTag theElectronCollection;
   edm::InputTag theMuonCollection;
   double thePtThreshold;
};
#endif
