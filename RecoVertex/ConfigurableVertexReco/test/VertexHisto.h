#ifndef RecoVertex_VertexHisto
#define RecoVertex_VertexHisto

#include <string>
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
// #include "RecoVertex/ConfigurableVertexReco/test/TrackHisto.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"                                  

class VertexHisto {
  /**
   *  Vertex histogramming.
   */
   public:
      VertexHisto( const std::string & filename="vertices.root",
                   const std::string & trackname="tracks.root" );
      ~VertexHisto();
      void analyse ( const TrackingVertex & sim, const TransientVertex & rec,
                     const std::string & name ) const;
      void saveTracks ( const TransientVertex & rec, 
                        const reco::RecoToSimCollection & p,
                        const std::string & name ) const;
   
   private:
      void stamp();

   private:
      std::string filename_;
   //   TrackHisto tracks_;
      bool hasStamped;
};

#endif
