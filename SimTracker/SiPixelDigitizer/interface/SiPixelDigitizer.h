#ifndef SiPixelDigitizer_h
#define SiPixelDigitizer_h

/** \class SiPixelDigitizer
 *
 * SiPixelDigitizer produces digis from SimHits
 * The real algorithm is in SiPixelDigitizerAlgorithm
 *
 * \author Michele Pioppi-INFN Perugia
 *
 * \version   Sep 26 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizerAlgorithm.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
//#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLinkCollection.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace cms
{
  class SiPixelDigitizer : public edm::EDProducer 
  {
  public:

    explicit SiPixelDigitizer(const edm::ParameterSet& conf);

    virtual ~SiPixelDigitizer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

    virtual void beginJob() {}
  private:
    edm::ParameterSet conf_;
    bool first;
    SiPixelDigitizerAlgorithm*  _pixeldigialgo;
    typedef std::vector<std::string> vstring;
    vstring trackerContainers;
    typedef std::map<unsigned int, std::vector<PSimHit>,std::less<unsigned int> > simhit_map;
    typedef simhit_map::iterator simhit_map_iterator;
    simhit_map SimHitMap;
    std::vector<edm::DetSet<PixelDigi> > theDigiVector;
    std::vector<edm::DetSet<PixelDigiSimLink> > theDigiLinkVector;
    std::string geometryType;
    CLHEP::HepRandomEngine* rndEngine;
    //   std::vector<PixelDigiSimLink> linkcollector;

    // infrastructure to reject dead pixels as defined in db (added by F.Blekman)
  };
}


#endif
