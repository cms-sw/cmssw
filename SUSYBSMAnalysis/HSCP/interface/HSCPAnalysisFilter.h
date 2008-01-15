#ifndef SUSYBSMANALYSIS_HSCPAnalysisFilter_HSCPAnalysisFilter_H
#define SUSYBSMANALYSIS_HSCPAnalysisFilter_HSCPAnalysisFilter_H
//
// class declaration
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HSCPAnalysisFilter : public edm::EDFilter {
   public:
      explicit HSCPAnalysisFilter(const edm::ParameterSet&);
      ~HSCPAnalysisFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

 
      // ----------member data ---------------------------
};

#endif

