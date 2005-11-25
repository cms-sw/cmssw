#ifndef SiPixelDigitizer_h
#define SiPixelDigitizer_h

/** \class SiPixelDigitizer
 *
 * SiPixelDigitizer is the EDProducer subclass which clusters
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
#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizerAlgorithm.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"


namespace cms
{
  class SiPixelDigitizer : public edm::EDProducer 
  {
  public:

    explicit SiPixelDigitizer(const edm::ParameterSet& conf);

    virtual ~SiPixelDigitizer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ParameterSet conf_;
    SiPixelDigitizerAlgorithm  _pixeldigialgo;
    int irandom1,irandom2,irandom3;
    float frandom3,frandom4,frandom5;
    float angrandom1,angrandom2;
    float xexrand,xentrand, yexrand,yentrand, zexrand,zentrand;
    std::vector<PSimHit> thePixelHits;
    std::vector<PSimHit> detPixelHits;
    std::vector<PixelDigi> collector;
  };
}


#endif
