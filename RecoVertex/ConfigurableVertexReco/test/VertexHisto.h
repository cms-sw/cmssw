#ifndef RecoVertex_VertexHisto
#define RecoVertex_VertexHisto

#include <string>
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

class VertexHisto {
  /**
   *  Vertex histogramming.
   */
   public:
      VertexHisto( const std::string & filename="vertices.root" );
      ~VertexHisto();
      void analyse ( const TrackingVertex & sim, const TransientVertex & rec,
                     const std::string & name ) const;

      void stamp();

   private:
      std::string filename_;
      bool hasStamped;
};

#endif
