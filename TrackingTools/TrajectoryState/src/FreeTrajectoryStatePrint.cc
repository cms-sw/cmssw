#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include <iostream>

using namespace std;

ostream& operator<<(ostream& os, const FreeTrajectoryState& fts) {
  os << "parameters" << endl;
  { 
    AlgebraicVector v = fts.parameters().vector();
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
  if (fts.hasError()) { 
    os << "error" << endl;
    { 
      AlgebraicSymMatrix m = fts.curvilinearError().matrix();
      for (int i = 0; i < 5; i++) { 
	for (int j = 0; j < 5; j++) {
	  os.precision(6); os.width(13); os<<m[i][j];
	}
	os << endl;
      }
    }
  }
  else {
    os << "no error defined." << endl;
  }
  return os;
}
