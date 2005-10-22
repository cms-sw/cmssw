#ifndef SiStripDigitizer_h
#define SiStripDigitizer_h

/** \class SiStripDigitizer
 *
 * SiStripDigitizer is the EDProducer subclass which clusters
 * SiStripDigi/interface/StripDigi.h to SiStripCluster/interface/SiStripCluster.h
 *
 * \author Michele Pioppi-INFN Perugia
 *
 * \version   Sep 26 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizerAlgorithm.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"



namespace cms
{
  class SiStripDigitizer : public edm::EDProducer
  {
  public:

    explicit SiStripDigitizer(const edm::ParameterSet& conf);

    virtual ~SiStripDigitizer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ParameterSet conf_;
    SiStripDigitizerAlgorithm  _stripdigialgo;
    int irandom1,irandom2,irandom3;
    float frandom3,frandom4,frandom5;
    float angrandom1,angrandom2;
    float xexrand,xentrand, yexrand,yentrand, zexrand,zentrand;
    std::vector<PSimHit*> pseudoHitSingleContainer;

  };
}


#endif
