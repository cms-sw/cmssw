#ifndef SimG4CMS_HFDarkening_h
#define SimG4CMS_HFDarkening_h

class HFDarkening
{
public:
  HFDarkening();
  ~HFDarkening();
  
  float dose(int layer, float radius);
  float int_lumi(float intlumi);
  float degradation(float mrad);
  
private:
  float radius[13];
  float dose_layer_radius[33][13];
};

#endif // HFDarkening_h
