#ifndef SiStripDigitizer_h
#define SiStripDigitizer_h

/** \class SiStripDigitizer
 *
 *
 * \author Andrea Giammanco
 *

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizerAlgorithm.h"

namespace cms
{
  class SiStripDigitizer : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit SiStripDigitizer(const edm::ParameterSet& conf);

    virtual ~SiStripDigitizer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    SiStripDigitizerAlgorithm stripDigitizer_;
    edm::ParameterSet conf_;
    int irandom1,irandom2,irandom3;
    float frandom1,frandom2,frandom3,frandom4,frandom5;
    float angrandom1,angrandom2;
    float xexrand,xentrand, yexrand,yentrand, zexrand,zentrand;
    std::vector<PSimHit*> pseudoHitSingleContainer; // temporary! to be removed...
    std::vector<PSimHit> theStripHits;

    int numStrips;    // number of strips in the module
  };
}


#endif
