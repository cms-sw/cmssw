#ifndef RecoVertex_CVRAnalysis
#define RecoVertex_CVRAnalysis

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "RecoVertex/ConfigurableVertexReco/test/VertexHisto.h"

class CVRAnalysis : public edm::EDAnalyzer {
  /**
   *  Class that glues the combined btagging algorithm to the framework
   */
   public:
      explicit CVRAnalysis( const edm::ParameterSet & );
      ~CVRAnalysis();

      virtual void analyze( const edm::Event &, const edm::EventSetup &);

   private:
      void discussPrimary( const edm::Event & ) const;

   private:
      ConfigurableVertexReconstructor * vrec_;
      std::string trackcoll_;
      std::string vertexcoll_;
      std::string beamspot_;
      edm::InputTag trackingtruth_;
      std::string associator_;
      VertexHisto histo_;
      VertexHisto bhisto_;
};

#endif
