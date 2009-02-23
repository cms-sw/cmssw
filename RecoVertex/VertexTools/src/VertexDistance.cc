#include "RecoVertex/VertexTools/interface/VertexDistance.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cfloat>


using namespace reco;

Measurement1D VertexDistance::distance(const VertexState & vtx1, 
					 const VertexState & vtx2) const
{
  return distance(vtx1.position(), vtx1.error(),
  		  vtx2.position(), vtx2.error());
}

Measurement1D VertexDistance::distance(const Vertex & vtx1, 
					 const VertexState & vtx2) const
{
  return distance(GlobalPoint(Basic3DVector<float> (vtx1.position())), 
		  GlobalError(vtx1.covariance()),
  		  vtx2.position(), vtx2.error());
}


Measurement1D VertexDistance::distance(const VertexState & vtx1, 
					 const Vertex & vtx2) const
{
  return distance(vtx1.position(), vtx1.error(),
  		  GlobalPoint(Basic3DVector<float> (vtx2.position())), 
		  GlobalError(vtx2.covariance()));
}


Measurement1D 
VertexDistance::distance(const Vertex & vtx1, const Vertex & vtx2) const
{
  return distance(GlobalPoint(Basic3DVector<float> (vtx1.position())), 
		  GlobalError(vtx1.covariance()),
  		  GlobalPoint(Basic3DVector<float> (vtx2.position())), 
		  GlobalError(vtx2.covariance()));
}


float VertexDistance::compatibility(const VertexState & vtx1, 
				      const VertexState & vtx2) const
{
  return compatibility(vtx1.position(), vtx1.error(),
		       vtx2.position(), vtx2.error());
}

float VertexDistance::compatibility(const Vertex & vtx1, 
				      const VertexState & vtx2) const
{
  return compatibility(GlobalPoint(Basic3DVector<float> (vtx1.position())), 
		       GlobalError(vtx1.covariance()),
		       vtx2.position(), vtx2.error());
}

float VertexDistance::compatibility(const VertexState & vtx1, 
				      const Vertex & vtx2) const
{
  return compatibility(vtx1.position(), vtx1.error(),
  		  GlobalPoint(Basic3DVector<float> (vtx2.position())), 
		  GlobalError(vtx2.covariance()));
}


float VertexDistance::compatibility(const Vertex & vtx1, 
				      const Vertex & vtx2) const
{
  return compatibility(GlobalPoint(Basic3DVector<float> (vtx1.position())), 
		       GlobalError(vtx1.covariance()),
		       GlobalPoint(Basic3DVector<float> (vtx2.position())), 
		       GlobalError(vtx2.covariance()));
}
