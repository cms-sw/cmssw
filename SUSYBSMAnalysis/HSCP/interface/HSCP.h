#ifndef SUSYBSMANALYSIS_HSCP_HSCP_H
#define SUSYBSMANALYSIS_HSCP_HSCP_H
//
// class declaration
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HSCP : public edm::EDFilter {
   public:
      explicit HSCP(const edm::ParameterSet&);
      ~HSCP();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
};

#endif

