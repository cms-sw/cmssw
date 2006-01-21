#ifndef CaloSimAlgos_CaloVNoisifier_h
#define CaloSimAlgos_CaloVNoisifier_h

/**
   \class CaloVNoisifier

   \brief adds noise to the given frame

*/

class CaloSamples;

class CaloVNoisifier
{
public:
  virtual void noisify(CaloSamples & frame) const=0;
};

#endif
