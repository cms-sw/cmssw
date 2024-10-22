//-------------------------------------------------
//
//   Class: DTAngleParam.cc
//
//   Description:
//       Provide time_vs_x_b functions for different angle values
//
//
//   Author List:
//   P. Ronchese           INFN - PADOVA
//   Modifications:
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DTAngleParam.h"

//-------------
// C Headers --
//-------------
#include <cmath>

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//----------------
// Constructors --
//----------------

DTAngleParam::DTAngleParam()
    :  // default constructor : assume 0 angle
      _angle(0.0),
      l_func(9),
      h_func(10) {}

DTAngleParam::DTAngleParam(float angle) {
  // determine angle bin in -45/+45 range by steps of 5

  int i_ang = static_cast<int>(angle + 45.0);
  int l_bin;

  if (i_ang < 0)
    l_bin = 0;
  else if (i_ang >= 90)
    l_bin = 17;
  else
    l_bin = i_ang / 5;

  int h_bin = l_bin + 1;

  // store actual angle and function parameters for lower/higher bin bounds
  _angle = angle;
  l_func = ParamFunc(l_bin);
  h_func = ParamFunc(h_bin);
}

//--------------
// Destructor --
//--------------

DTAngleParam::~DTAngleParam() {}

//--------------
// Operations --
//--------------

float DTAngleParam::time(float bwire, float xcoor) const {
  // compute drift time for lower/higher bin bounds
  float l_time = l_func.time(bwire, xcoor);
  float h_time = h_func.time(bwire, xcoor);

  // interpolate times for actual angle
  return l_time + ((h_time - l_time) * l_func.dist(_angle) / l_func.dist(h_func));
}

const int DTAngleParam::table_num_terms[19] = {7, 7, 6, 7, 8, 7, 5, 6, 5, 7, 4, 2, 7, 8, 6, 6, 6, 8, 7};

const int DTAngleParam::table_pow_field[190] = {
    0,  0,  0,  0,  1,  1,  1,  -1, -1, -1, 0,  0,  0,  0,  1,  1,  1,  -1, -1, -1, 0,  0,  0,  0,  2,  2,  -1, -1,
    -1, -1, 0,  0,  0,  1,  1,  0,  1,  -1, -1, -1, 0,  0,  0,  0,  2,  1,  1,  1,  -1, -1, 0,  0,  3,  0,  0,  2,
    1,  -1, -1, -1, 0,  0,  1,  1,  1,  -1, -1, -1, -1, -1, 0,  0,  0,  0,  1,  1,  -1, -1, -1, -1, 0,  0,  1,  1,
    1,  -1, -1, -1, -1, -1, 0,  0,  0,  2,  0,  3,  1,  -1, -1, -1, 0,  0,  0,  0,  -1, -1, -1, -1, -1, -1, 0,  0,
    -1, -1, -1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  2,  1,  3,  -1, -1, -1, 0,  0,  0,  0,  3,  1,  1,  2,  -1, -1,
    0,  0,  0,  1,  0,  1,  -1, -1, -1, -1, 0,  0,  0,  0,  1,  1,  -1, -1, -1, -1, 0,  0,  0,  0,  1,  1,  -1, -1,
    -1, -1, 0,  0,  0,  0,  1,  1,  1,  2,  -1, -1, 0,  0,  0,  0,  1,  1,  1,  -1, -1, -1};

const int DTAngleParam::table_pow_xcoor[190] = {
    0,  1,  3,  2,  0,  1,  2,  -1, -1, -1, 0,  1,  3,  2,  0,  1,  2,  -1, -1, -1, 0,  1,  3,  2,  0,  1,  -1, -1,
    -1, -1, 0,  1,  3,  0,  1,  2,  2,  -1, -1, -1, 0,  1,  3,  2,  1,  0,  1,  2,  -1, -1, 0,  1,  0,  3,  2,  1,
    0,  -1, -1, -1, 0,  1,  0,  1,  2,  -1, -1, -1, -1, -1, 0,  1,  3,  2,  0,  2,  -1, -1, -1, -1, 0,  1,  0,  1,
    2,  -1, -1, -1, -1, -1, 0,  1,  3,  1,  2,  0,  2,  -1, -1, -1, 0,  1,  3,  2,  -1, -1, -1, -1, -1, -1, 0,  1,
    -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  3,  2,  0,  2,  0,  -1, -1, -1, 0,  1,  3,  2,  0,  0,  2,  0,  -1, -1,
    0,  1,  3,  0,  2,  2,  -1, -1, -1, -1, 0,  1,  3,  2,  0,  1,  -1, -1, -1, -1, 0,  1,  3,  2,  0,  1,  -1, -1,
    -1, -1, 0,  1,  3,  2,  0,  1,  2,  0,  -1, -1, 0,  1,  3,  2,  0,  1,  2,  -1, -1, -1};

const float DTAngleParam::table_offsc[19] = {-1.6341677E-03,
                                             -1.9150710E-03,
                                             -1.9032805E-03,
                                             -1.9542247E-03,
                                             -1.9582238E-03,
                                             -1.8554466E-03,
                                             -1.9125413E-03,
                                             -1.9741140E-03,
                                             -1.9815513E-03,
                                             -1.9835865E-03,
                                             -1.9833876E-03,
                                             -1.8502035E-03,
                                             -1.9108272E-03,
                                             -1.9580512E-03,
                                             -1.9557050E-03,
                                             -1.8146527E-03,
                                             -1.7686716E-03,
                                             -1.7501116E-03,
                                             -1.6865443E-03};

const float DTAngleParam::table_coeff[190] = {0.32362,
                                              -0.10029,
                                              0.33372E-01,
                                              -0.97290E-01,
                                              0.14761,
                                              -0.15878,
                                              0.43523E-01,
                                              0,
                                              0,
                                              0,

                                              0.32860,
                                              -0.11620,
                                              0.25506E-01,
                                              -0.75019E-01,
                                              0.11910,
                                              -0.98917E-01,
                                              0.18437E-01,
                                              0,
                                              0,
                                              0,

                                              0.34235,
                                              -0.14353,
                                              0.17375E-01,
                                              -0.48856E-01,
                                              0.27060,
                                              -0.14935,
                                              0,
                                              0,
                                              0,
                                              0,

                                              0.34117,
                                              -0.15159,
                                              0.11491E-01,
                                              0.89598E-01,
                                              -0.67190E-01,
                                              -0.32898E-01,
                                              0.10769E-01,
                                              0,
                                              0,
                                              0,

                                              0.34218,
                                              -0.14871,
                                              0.99750E-02,
                                              -0.31434E-01,
                                              0.41733E-01,
                                              0.76480E-01,
                                              -0.90189E-01,
                                              0.21807E-01,
                                              0,
                                              0,

                                              0.34941,
                                              -0.15649,
                                              0.31413,
                                              0.99644E-02,
                                              -0.29251E-01,
                                              -0.56078E-01,
                                              0.11125E-01,
                                              0,
                                              0,
                                              0,

                                              0.35068,
                                              -0.17724,
                                              0.59457E-01,
                                              -0.73007E-01,
                                              0.24587E-01,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,

                                              0.35073,
                                              -0.16944,
                                              0.46646E-02,
                                              -0.12188E-01,
                                              0.26793E-01,
                                              -0.90753E-02,
                                              0,
                                              0,
                                              0,
                                              0,

                                              0.35095,
                                              -0.17652,
                                              0.34244E-01,
                                              -0.25854E-01,
                                              0.33369E-02,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,

                                              0.35302,
                                              -0.17322,
                                              0.19561E-02,
                                              -0.59183E-01,
                                              -0.63050E-02,
                                              0.16520,
                                              0.62952E-02,
                                              0,
                                              0,
                                              0,

                                              0.34995,
                                              -0.16756,
                                              0.35931E-02,
                                              -0.11169E-01,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,

                                              0.34867,
                                              -0.17543,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,

                                              0.34773,
                                              -0.16523,
                                              0.46184E-02,
                                              -0.14212E-01,
                                              -0.11195,
                                              0.60685E-02,
                                              0.20589,
                                              0,
                                              0,
                                              0,

                                              0.34647,
                                              -0.16158,
                                              0.57246E-02,
                                              -0.17762E-01,
                                              -0.41158,
                                              -0.41086E-01,
                                              0.72317E-02,
                                              0.17672,
                                              0,
                                              0,

                                              0.34311,
                                              -0.15890,
                                              0.67728E-02,
                                              -0.27006E-01,
                                              -0.20412E-01,
                                              0.77523E-02,
                                              0,
                                              0,
                                              0,
                                              0,

                                              0.33889,
                                              -0.14902,
                                              0.10912E-01,
                                              -0.32543E-01,
                                              -0.47053E-01,
                                              0.26043E-01,
                                              0,
                                              0,
                                              0,
                                              0,

                                              0.33616,
                                              -0.13829,
                                              0.15988E-01,
                                              -0.47280E-01,
                                              -0.71785E-01,
                                              0.40111E-01,
                                              0,
                                              0,
                                              0,
                                              0,

                                              0.33327,
                                              -0.13235,
                                              0.23152E-01,
                                              -0.62435E-01,
                                              -0.11302,
                                              0.10277,
                                              -0.30199E-01,
                                              0.52548E-01,
                                              0,
                                              0,

                                              0.33239,
                                              -0.14388,
                                              0.22312E-01,
                                              -0.55883E-01,
                                              -0.95301E-01,
                                              0.92658E-01,
                                              -0.21228E-01,
                                              0,
                                              0,
                                              0};

DTAngleParam::ParamFunc::ParamFunc() {
  /*
   // default constructor: assume central bin ( 0 angle )
   bin_angle = 0.0;
   num_terms = table_num_terms+ 9;
   pow_field = table_pow_field+ 90;
   pow_xcoor = table_pow_xcoor+ 90;
   offsc     = table_offsc    + 9;
   coeff     = table_coeff    + 90;
  */
}

DTAngleParam::ParamFunc::ParamFunc(int bin) {
  // store bound
  bin_angle = (bin * 5.0) - 45.0;
  // select parameters inside tables
  num_terms = table_num_terms + bin;
  pow_field = table_pow_field + (bin * 10);
  pow_xcoor = table_pow_xcoor + (bin * 10);
  offsc = table_offsc + bin;
  coeff = table_coeff + (bin * 10);
}

/*
void DTAngleParam::ParamFunc::set(int bin) {

 // reset bound
 bin_angle = (bin*5.0)-45.0;
 // select parameters inside tables
 num_terms = table_num_terms+ bin;
 pow_field = table_pow_field+(bin*10);
 pow_xcoor = table_pow_xcoor+(bin*10);
 offsc     = table_offsc    + bin;
 coeff     = table_coeff    +(bin*10);

}
*/

DTAngleParam::ParamFunc::~ParamFunc() {}

float DTAngleParam::ParamFunc::time(float bwire, float xcoor) const {
  // compute drift time

  // set constant offset
  float y = *offsc;

  // build up polynomial
  int i;
  // int j;
  // float x;
  for (i = 0; i < *num_terms; i++) {
    //  x = 1.0;
    //  for ( j = 0 ; j < pow_field[i] ; j++ ) x *= bwire;
    //  for ( j = 0 ; j < pow_xcoor[i] ; j++ ) x *= xcoor;
    //  y += (*coeff)*x;
    y += coeff[i] * pow(bwire, pow_field[i]) * pow(xcoor, pow_xcoor[i]);
  }

  return y;
}
