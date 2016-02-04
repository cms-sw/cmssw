#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include <cstdio>

using namespace std;

  HcalTBEventPosition::HcalTBEventPosition() :
    hfTableX_(-1e24),
    hfTableY_(-1e24),
    hfTableV_(-1e24),
    hbheTableEta_(-1000),
    hbheTablePhi_(-1000),
    ax_(),
    ay_(),
    bx_(),
    by_(),
    cx_(),
    cy_(),
    dx_(),
    dy_(),
    ex_(),
    ey_(),
    fx_(),
    fy_(),
    gx_(),
    gy_(),
    hx_(),
    hy_() {
  }

  void HcalTBEventPosition::getChamberHits ( char chamberch,
					     vector<double>& xvec,
					     vector<double>& yvec ) const {
    switch (chamberch) {
    case 'A': xvec = ax_; yvec = ay_; break;
    case 'B': xvec = bx_; yvec = by_; break;
    case 'C': xvec = cx_; yvec = cy_; break;
    case 'D': xvec = dx_; yvec = dy_; break;
    case 'E': xvec = ex_; yvec = ey_; break;
    case 'F': xvec = fx_; yvec = fy_; break;
    case 'G': xvec = gx_; yvec = gy_; break;
    case 'H': xvec = hx_; yvec = hy_; break;
    default:
      cerr << "Unrecognized chamber character " << chamberch << endl;
      return;
    }
  }

  void HcalTBEventPosition::setChamberHits ( char chamberch,
					     const vector<double>& xvec,
					     const vector<double>& yvec ) {
    switch (chamberch) {
    case 'A': ax_ = xvec; ay_ = yvec; break;
    case 'B': bx_ = xvec; by_ = yvec; break;
    case 'C': cx_ = xvec; cy_ = yvec; break;
    case 'D': dx_ = xvec; dy_ = yvec; break;
    case 'E': ex_ = xvec; ey_ = yvec; break;
    case 'F': fx_ = xvec; fy_ = yvec; break;
    case 'G': gx_ = xvec; gy_ = yvec; break;
    case 'H': hx_ = xvec; hy_ = yvec; break;
    default:
      cerr << "Unrecognized chamber character " << chamberch << endl;
      return;
    }
  }

  void HcalTBEventPosition::setHFtableCoords( double x, double y, double v ) {
    hfTableX_ = x;
    hfTableY_ = y;
    hfTableV_ = v;
  }

  void HcalTBEventPosition::setHBHEtableCoords( double eta, double phi ) {
    hbheTableEta_ = eta;
    hbheTablePhi_ = phi;
  }

  ostream& operator<<(ostream& s, const HcalTBEventPosition& htbep) {
    char str[180];

    snprintf (str,180, "HF Table (X,Y,V) = (%f,%f,%f)\n",
	     htbep.hfTableX(),htbep.hfTableY(),htbep.hfTableV());
    s << str;

    snprintf (str, 180, "HB/HE Table (eta,phi) = (%f,%f)\n",
	     htbep.hbheTableEta(),htbep.hbheTablePhi());
    s << str;

    vector<double> xvec, yvec;
    vector<double>::const_iterator j;

    htbep.getChamberHits('A', xvec, yvec);

    s << "WC Ax: ";
    for (j=xvec.begin(); j!=xvec.end(); j++) {
      if (j!=xvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    s << "WC Ay: ";
    for (j=yvec.begin(); j!=yvec.end(); j++) {
      if (j!=yvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    htbep.getChamberHits('B', xvec, yvec);

    s << "WC Bx: ";
    for (j=xvec.begin(); j!=xvec.end(); j++) {
      if (j!=xvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    s << "WC By: ";
    for (j=yvec.begin(); j!=yvec.end(); j++) {
      if (j!=yvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    htbep.getChamberHits('C', xvec, yvec);

    s << "WC Cx: ";
    for (j=xvec.begin(); j!=xvec.end(); j++) {
      if (j!=xvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    s << "WC Cy: ";
    for (j=yvec.begin(); j!=yvec.end(); j++) {
      if (j!=yvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    htbep.getChamberHits('D', xvec, yvec);

    s << "WC Dx: ";
    for (j=xvec.begin(); j!=xvec.end(); j++) {
      if (j!=xvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    s << "WC Dy: ";
    for (j=yvec.begin(); j!=yvec.end(); j++) {
      if (j!=yvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    htbep.getChamberHits('E', xvec, yvec);

    s << "WC Ex: ";
    for (j=xvec.begin(); j!=xvec.end(); j++) {
      if (j!=xvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    s << "WC Ey: ";
    for (j=yvec.begin(); j!=yvec.end(); j++) {
      if (j!=yvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    htbep.getChamberHits('F', xvec, yvec);

    s << "WC Fx: ";
    for (j=xvec.begin(); j!=xvec.end(); j++) {
      if (j!=xvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    s << "WC Fy: ";
    for (j=yvec.begin(); j!=yvec.end(); j++) {
      if (j!=yvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    htbep.getChamberHits('G', xvec, yvec);

    s << "WC Gx: ";
    for (j=xvec.begin(); j!=xvec.end(); j++) {
      if (j!=xvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    s << "WC Gy: ";
    for (j=yvec.begin(); j!=yvec.end(); j++) {
      if (j!=yvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    htbep.getChamberHits('H', xvec, yvec);

    s << "WC Hx: ";
    for (j=xvec.begin(); j!=xvec.end(); j++) {
      if (j!=xvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    s << "WC Hy: ";
    for (j=yvec.begin(); j!=yvec.end(); j++) {
      if (j!=yvec.begin()) s << ", ";
      s << *j;
    }
    s << endl;

    return s;
  }
