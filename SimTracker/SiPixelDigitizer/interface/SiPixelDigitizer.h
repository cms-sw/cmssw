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
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLinkCollection.h"

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
    std::vector<PSimHit> thePixelHits;
    std::vector<PixelDigi> collector;
    typedef std::map<unsigned int, std::vector<PSimHit>,std::less<unsigned int> > simhit_map;
    typedef simhit_map::iterator simhit_map_iterator;
    simhit_map SimHitMap;
    std::vector<PixelDigiSimLink> linkcollector;
  };
}


#endif
