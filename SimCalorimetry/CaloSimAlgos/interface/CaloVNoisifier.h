#ifndef CaloVNoisifier_h
#define CaloVNoisifier_h

/**
   \class CaloVNoisifier

   \brief adds noise to the given frame

*/
namespace CLHEP {
  class RandGaussQ;
}

class CaloSamples;

namespace cms {

  class CaloVNoisifier
  {
  public:
    CaloVNoisifier();
    virtual ~CaloVNoisifier();

    virtual void noisify(CaloSamples & frame) const=0;

  protected:
    CLHEP::RandGaussQ * theRandGaussian;
  };
}

#endif
