#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include <ostream>

using namespace std;

ostream& operator<<(std::ostream& os, const TrajectoryStateOnSurface& tsos) {
  os << "global parameters" << endl;
  {
    const AlgebraicVector6 &v = tsos.globalParameters().vector();
    os << "x = ";
    {
      for (int i = 0; i < 3; i++) {
        os.precision(6); os.width(13); os<<v[i];
      }
    }
    os << endl;
    os << "p = ";
    {
      for (int i = 3; i < 6; i++) {
        os.precision(6); os.width(13); os<<v[i];
      }
    }
    os << endl;
  }
  if ( tsos.hasError()) {
    os << "global error" << endl;
    {
      const AlgebraicSymMatrix55 &m = tsos.curvilinearError().matrix();
      for (int i = 0; i < 5; i++) {
	for (int j = 0; j < 5; j++) {
	  os.precision(6); os.width(13); os<<m(i,j);
	}
	os << endl;
      }
    }
  }
  if ( tsos.localParameters().charge()!=0 )
    os << "local parameters (q/p,v',w',v,w)" << endl;
  else
    os << "local parameters for neutral (1/p,v',w',v,w)" << endl;
  {
    const AlgebraicVector5 &v = tsos.localParameters().mixedFormatVector();
    for (int i = 0; i < 5; i++) {
      os.precision(6); os.width(13); os<<v[i];
    }
    os << endl;
  }
  if ( tsos.hasError()) {
    os << "local error" << endl;
    {
      const AlgebraicSymMatrix55 &m = tsos.localError().matrix();
      for (int i = 0; i < 5; i++) {
	for (int j = 0; j < 5; j++) {
	  os.precision(6); os.width(13); os<<m(i,j);
	}
	os << endl;
      }
    }
  }
  os << "Defined at ";
  if ( tsos.surfaceSide()==SurfaceSideDefinition::beforeSurface )  os << "beforeSurface";
  else if ( tsos.surfaceSide()==SurfaceSideDefinition::afterSurface )  os << "afterSurface";
  else  os << "atCenterOfSurface";
  os << endl;

  // magnetic field
  os << "Magnetic field in inverse GeV: " << tsos.globalParameters().magneticFieldInInverseGeV(tsos.globalPosition());
  os <<endl;

  return os;
}
