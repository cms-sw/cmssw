#ifndef CaloSimAlgos_CaloVShape_h
#define CaloSimAlgos_CaloVShape_h 1

/**

   \class CaloVShape

   \brief Electronic response of the preamp
*/

class CaloVShape 
{
 public:

  CaloVShape() {}
  virtual ~CaloVShape() {}

  virtual double       operator () (double) const = 0 ;
  virtual double       timeToRise()         const = 0 ;

 protected:

 private:
};

#endif
