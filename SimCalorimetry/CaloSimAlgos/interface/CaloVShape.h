#ifndef CaloSimAlgos_CaloVShape_h
#define CaloSimAlgos_CaloVShape_h 1

/**

   \class CaloVShape

   \brief Electronic response of the preamp
*/

class CaloVShape 
{
public:
  CaloVShape() : tpeak_(0.) {}
  virtual ~CaloVShape(){}

  virtual double operator () (double) const=0;

  //virtual double derivative (double) const = 0;

  double getTpeak () const{return tpeak_;}

  void setTpeak (double value){tpeak_=value;}

private:

  double tpeak_;
};

#endif
