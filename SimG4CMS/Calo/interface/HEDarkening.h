#ifndef SimG4CMS_HEDarkening_h
#define SimG4CMS_HEDarkening_h
//
// Simple class with parameterizaed function to get darkening attenuation 
// coefficiant for SLHC conditions
// = degradation(int_lumi(intlumi) * dose(layer,Radius)), where
// intlumi is integrated luminosity (fb-1), 
// layer is HE layer number (from -1 up// to 17), NB: 1-19 in HcalTestNumbering
// Radius is radius from the beam line (cm) 
//

class HEDarkening {

public:
  HEDarkening();
  ~HEDarkening();

  float dose(int layer,float radius);
  float int_lumi(float intlumi);
  float degradation(float mrad);

private:
  float radius[24];
  float dose_lm1_l0[24];
  float dose_l1_l3[24];
  float dose_l4_l5[24];
  float dose_l6_l8[24];
  float dose_l9_l10[24];
  float dose_l11_l13[24];
  float dose_l14_l15[24];
  float dose_l16_l17[24];
};


#endif // HEDarkening_h
