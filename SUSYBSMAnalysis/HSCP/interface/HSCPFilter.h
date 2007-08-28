#ifndef SUSYBSMANALYSIS_HSCPFilter_HSCPFilter_H
#define SUSYBSMANALYSIS_HSCPFilter_HSCPFilter_H
//
// class declaration
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HSCPFilter : public edm::EDFilter {
   public:
      explicit HSCPFilter(const edm::ParameterSet&);
      ~HSCPFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      float m_dedxCut1;
      float m_tkPCut1;
      float m_dedxCut2;
      float m_tkPCut2;
      float m_dedxCut3;
      float m_tkPCut3;
      float m_singleMuPtMin;
      float m_doubleMuPtMin;
 
      // ----------member data ---------------------------
};

#endif

