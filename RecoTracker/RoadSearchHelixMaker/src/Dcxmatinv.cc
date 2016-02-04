//babar #include "BaBar/BaBar.hh"
//babar #include <math.h>
#include <cmath>
#include <iostream>
//babar #include "DcxReco/Dcxmatinv.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/Dcxmatinv.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using std::cout;
using std::endl;

extern int Dcxmatinv(double *array, int *norder, double *det){
  /* System generated locals */
  int i__3;
  double d__1;

  /* Local variables */
  const int nmax = 10;
//  edm::LogInfo("RoadSearch") << "norder in Dcxmatinv = " << *norder ;
  if (*norder > nmax){
    edm::LogInfo("RoadSearch") << "In Dcxmatinv, norder ( = " << *norder << " ) > nmax ( = "
			   << nmax << " ); error" ; return 1000;
  }
  static double amax, save;
  static int i, j, k, l, ik[nmax], jk[nmax];

  /* Parameter adjustments */
  array -= (nmax+1);

  /* Function Body */
  *det = (double)1.;
  for (k = 1; k <= *norder; ++k) {

    /*       FIND LARGEST ELEMENT ARRAY(I, J) IN REST OF MATRIX */

    amax = (double)0.;
  L21:
    for (i = k; i <= *norder; ++i) {
      for (j = k; j <= *norder; ++j) {
	d__1 = array[i + j * nmax]; 
	if ((fabs(amax)-fabs(d__1)) <= 0.) {
	  amax = array[i + j * nmax];
	  ik[k - 1] = i;
	  jk[k - 1] = j;
	}
      }
    }

    /*       INTERCHANGE ROWS AND COLUMNS TO PUT AMAX IN ARRAY(K, K) */

    if (amax == 0.) {*det = (double)0.; return 1001;}

    i = ik[k - 1];
    if ((i__3 = i - k) < 0) {
      goto L21;
    } else if (i__3 == 0) {
      goto L51;
    } else {
      goto L43;
    }
  L43:
    for (j = 1; j <= *norder; ++j) {
      save = array[k + j * nmax];
      array[k + j * nmax] = array[i + j * nmax];
      array[i + j * nmax] = -save;
    }
  L51:
    j = jk[k - 1];
    if ((i__3 = j - k) < 0) {
      goto L21;
    } else if (i__3 == 0) {
      goto L61;
    } else {
      goto L53;
    }
  L53:
    for (i = 1; i <= *norder; ++i) {
      save = array[i + k * nmax];
      array[i + k * nmax] = array[i + j * nmax];
      array[i + j * nmax] = -save;
    }

    /*       ACCUMULATE ELEMENTS OF INVERSE MATRIX */

  L61:
    for (i = 1; i <= *norder; ++i) {
      if (i - k != 0) {
	array[i + k * nmax] = -array[i + k * nmax] / amax;
      }	
    }
    for (i = 1; i <= *norder; ++i) {
      for (j = 1; j <= *norder; ++j) {
	if (i - k != 0) {
	  goto L74;
	} else {
	  goto L80;
	}
      L74:
	if (j - k != 0) {
	  goto L75;
	} else {
	  goto L80;
	}
      L75:
	array[i+j*nmax] += array[i+k*nmax] * array[k+j*nmax];
      L80:
	;
      }
    }
    for (j = 1; j <= *norder; ++j) {
      if (j - k != 0) {
	array[k + j * nmax] /= amax;
      }
    }
    array[k + k * nmax] = (double)1. / amax;
    *det *= amax;
  }

  /*       RESTORE ORDERING OF MATRIX */

  for (l = 1; l <= *norder; ++l) {
    k = *norder - l + 1;
    j = ik[k - 1];
    if (j - k <= 0) {
      goto L111;
    } else {
      goto L105;
    }
  L105:
    for (i = 1; i <= *norder; ++i) {
      save = array[i + k * nmax];
      array[i + k * nmax] = -array[i + j * nmax];
      array[i + j * nmax] = save;
    }
  L111:
    i = jk[k - 1];
    if (i - k <= 0) {
      goto L130;
    } else {
      goto L113;
    }
  L113:
    for (j = 1; j <= *norder; ++j) {
      save = array[k + j * nmax];
      array[k + j * nmax] = -array[i + j * nmax];
      array[i + j * nmax] = save;
    }
  L130:
    ;
  }
  return 0;
} /* Dcxmatinv */
