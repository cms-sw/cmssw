#ifndef CaloVNoisifier_h
#define CaloVNoisifier_h

/**
   \class CaloVNoisifier

   \brief adds noise to the given frame

*/

class CaloSamples;

namespace cms {

  class CaloVNoisifier
  {
  public:
    virtual void noisify(CaloSamples & frame) const=0;
  };
}

#endif
