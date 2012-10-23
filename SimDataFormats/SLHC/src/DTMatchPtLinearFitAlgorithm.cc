#include "SimDataFormats/SLHC/interface/DTMatchPtAlgorithms.h" 

/*

(1, 0) --> 0
(2, 0) --> 1
(2, 1) --> 2
(3, 0) --> 3
(3, 1) --> 4
(3, 2) --> 5
(4, 0) --> 6
(4, 1) --> 7
(4, 2) --> 8
(4, 3) --> 9
(5, 0) --> 10
(5, 1) --> 11
(5, 2) --> 12
(5, 3) --> 13
(5, 4) --> 14

*/

const float DTMatchPtAlgorithms::_chi2[] = 
  {
    NAN, 
    14.0236, 14.3372, 
    27.9615, 32.9746, NAN,
    947.799, 705.204, 664.357, 280.836,
    761.801, 945.329, 1013.12, 434.828, NAN
  };

const float DTMatchPtAlgorithms::_p0[] = 
  {
    NAN,
    3.41E-05, 2.39E-05,
    2.25E-05, 2.22E-05, NAN,
    -6.64E-05, -7.19E-05, -5.75E-05, -0.39E-05,
    -6.05E-05, -7.92E-05, -4.56E-05, -0.54E-05, NAN
  };

const float DTMatchPtAlgorithms::_e0[] = 
  {
    NAN,
    1.25E-05, 0.90E-05,
    1.36E-05, 1.18E-05, NAN,
    0.70E-05, 0.59E-05, 0.43E-05, 0.62E-05,
    0.67E-05, 0.57E-05, 0.44E-05, 0.58E-05, NAN};

const float DTMatchPtAlgorithms::_p1[] = 
  {
    NAN,
    0.091614, 0.068779,
    0.114228, 0.091263, NAN,
    0.378154, 0.355883, 0.288844, 0.265144,
    0.401449, 0.379040, 0.311536, 0.287998, NAN
  };

const float DTMatchPtAlgorithms::_e1[] = 
  {
    NAN,
    0.000399, 0.000251,
    0.000354, 0.000249, NAN,
    0.000179, 0.000183, 0.000122, 0.000141,
    0.000175, 0.000171, 0.000132, 0.000132, NAN
  };


float DTMatchPtAlgorithms::chi2_linearfit(int L1, int L2) {
  // chi2 of dephi vs invPt linear fit 
  L1 = tracker_lay_Id_to_our(L1);
  L2 = tracker_lay_Id_to_our(L2);
  if(L1 > L2) {
    int idx = (L1*(L1-1))/2 + L2;
    return _chi2[idx];
  }
  else if(L2 > L1) {
    int idx = (L2*(L2-1))/2 + L1;
    return _chi2[idx];
  } 
  else return NAN;
}

float DTMatchPtAlgorithms::slope_linearfit(int L1, int L2) {
  // angular coefficient for dephi vs invPt linear fit
  L1 = tracker_lay_Id_to_our(L1);
  L2 = tracker_lay_Id_to_our(L2);
  if(L1 > L2) {
    int idx = (L1*(L1-1))/2 + L2;
    /*
    cout << "(" << L1 << ", " << L2 << ") --> idx = " << idx << ": slope = " 
	 <<  _p1[idx] << endl;
    */
    return _p1[idx];
  }
  else if(L2 > L1) {
    int idx = (L2*(L2-1))/2 + L1;
    /*
    cout << "(" << L1 << ", " << L2 << ") --> idx = " << idx << ": slope = " 
	 <<  _p1[idx] << endl;
    */
    return _p1[idx];
  } 
  else return NAN;
}

float DTMatchPtAlgorithms::sigma_slope_linearfit(int L1, int L2) {
  // sigma of angular coefficient for dephi vs invPt linear fit
  L1 = tracker_lay_Id_to_our(L1);
  L2 = tracker_lay_Id_to_our(L2);
  if(L1 > L2) {
    int idx = (L1*(L1-1))/2 + L2;
    return _e1[idx];
  }
  else if(L2 > L1) {
    int idx = (L2*(L2-1))/2 + L1;
    return _e1[idx];
  } 
  else return NAN;
}

float DTMatchPtAlgorithms::y_intercept_linearfit(int L1, int L2) {
  // dephi @ invPt=0
  L1 = tracker_lay_Id_to_our(L1);
  L2 = tracker_lay_Id_to_our(L2);
  if(L1 > L2) {
    int idx = (L1*(L1-1))/2 + L2;
    return _p0[idx];
  }
  else if(L2 > L1) {
    int idx = (L2*(L2-1))/2 + L1;
    return _p0[idx];
  } 
  else return NAN;
}

float DTMatchPtAlgorithms::sigma_y_intercept_linearfit(int L1, int L2) {
  // sigma of dephi @ invPt=0
  L1 = tracker_lay_Id_to_our(L1);
  L2 = tracker_lay_Id_to_our(L2);
  if(L1 > L2) {
    int idx = (L1*(L1-1))/2 + L2;
    return _e0[idx];
  }
  else if(L2 > L1) {
    int idx = (L2*(L2-1))/2 + L1;
    return _e0[idx];
  } 
  else return NAN;
}



