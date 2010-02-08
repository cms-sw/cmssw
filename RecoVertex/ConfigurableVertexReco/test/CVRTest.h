#ifndef RecoBTag_CVRTest
#define RecoBTag_CVRTest

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"

class CVRTest : public edm::EDAnalyzer {
  /**
   *  Class that glues the combined btagging algorithm to the framework
   */
   public:
      explicit CVRTest( const edm::ParameterSet & );
      ~CVRTest();

      virtual void analyze( const edm::Event &, const edm::EventSetup &);

   private:
      void discussPrimary( const edm::Event & ) const;

   private:
      ConfigurableVertexReconstructor * vrec_;
      std::string trackcoll_;
      std::string vertexcoll_;
      std::string beamspot_;

};

#endif
