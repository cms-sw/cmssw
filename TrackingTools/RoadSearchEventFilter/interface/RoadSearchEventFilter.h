#ifndef RoadSearchEventFilter_h
#define RoadSearchEventFilter_h


#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"


class RoadSearchEventFilter : public edm::EDFilter {
   public:
      explicit RoadSearchEventFilter(const edm::ParameterSet&);
      ~RoadSearchEventFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      unsigned int numberOfSeeds_;
      std::string seedCollectionLabel_; 

};

#endif
