#include "SimMuon/GEMDigitizer/src/GEMSynchronizer.h"
#include "SimMuon/GEMDigitizer/src/GEMSimSetUp.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "CLHEP/Random/RandGaussQ.h"


namespace 
{
  // "magic" parameter for cosmics
  const double COSMIC_PAR =  37.62;
}


GEMSynchronizer::GEMSynchronizer(const edm::ParameterSet& config):
  gauss1_(0), gauss2_(0)
{
  timeResolution_ = config.getParameter<double>("timeResolution");
  averageShapingTime_ = config.getParameter<double>("averageShapingTime");
  timeJitter_ = config.getParameter<double>("timeJitter");
  signalPropagationSpeed_ = config.getParameter<double>("signalPropagationSpeed");
  cosmics_ = config.getParameter<bool>("cosmics");
  bxwidth_ = config.getParameter<double>("bxwidth");

  // signal propagation speed in vacuum in [m/s]
  const double cspeed = 299792458;
  // signal propagation speed in material in [cm/ns]
  signalPropagationSpeed_ = signalPropagationSpeed_ * cspeed * 1e+2 * 1e-9;
}


void GEMSynchronizer::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  gauss1_ = new CLHEP::RandGaussQ(eng);
  gauss2_ = new CLHEP::RandGaussQ(eng);
}


GEMSynchronizer::~GEMSynchronizer()
{
  if (gauss1_) delete gauss1_;
  if (gauss2_) delete gauss2_;
}


int GEMSynchronizer::getSimHitBx(const PSimHit* simhit)
{
  GEMSimSetUp* simsetup = getGEMSimSetUp();
  const GEMGeometry * geometry = simsetup->getGeometry();
  // calibration offset for a particular detector part
  float calibrationTime = simsetup->getTime(simhit->detUnitId());

  int bx = -999;
  LocalPoint simHitPos = simhit->localPosition();
  float tof = simhit->timeOfFlight();
  // random Gaussian time correction due to electronics jitter
  float randomJitterTime = gauss1_->fire(0., timeJitter_);
  
  GEMDetId shdetid(simhit->detUnitId());

  const GEMEtaPartition* shRoll = 0;

  for(const auto &det: geometry->dets())
  {
    if( dynamic_cast< GEMEtaPartition* >( det ) != 0 )
    {
      GEMEtaPartition* roll = dynamic_cast< GEMEtaPartition* >( det );

      if(roll->id() == shdetid)
      {
        shRoll = roll;
        break;
      }
    }
  }

  if(shRoll != 0)
  {
    float distanceFromEdge = 0;
    float halfStripLength = 0.;

    if(shRoll->id().region() == 0)
    {
      throw cms::Exception("Geometry")
        << "GEMSynchronizer::getSimHitBx() - this GEM id is from barrel, which cannot happen:  "<<shRoll->id()<< "\n";
    }
    else
    {
      const TrapezoidalStripTopology* top = dynamic_cast<const TrapezoidalStripTopology*> (&(shRoll->topology()));
      halfStripLength = 0.5 * top->stripLength();
      distanceFromEdge = halfStripLength - simHitPos.y();
    }
    
    // average time for the signal to propagate from the SimHit to the top of a strip
    float averagePropagationTime =  distanceFromEdge/signalPropagationSpeed_;
    // random Gaussian time correction due to the finite timing resolution of the detector
    float randomResolutionTime = gauss2_->fire(0., timeResolution_);

    float simhitTime = tof + (averageShapingTime_ + randomResolutionTime) + (averagePropagationTime + randomJitterTime);
    float referenceTime = calibrationTime + halfStripLength/signalPropagationSpeed_ + averageShapingTime_;
    float timeDifference = cosmics_ ? (simhitTime - referenceTime)/COSMIC_PAR : simhitTime - referenceTime;

    // assign the bunch crossing
    bx = static_cast<int>( std::round((timeDifference)/bxwidth_) );

    // check time
    const bool debug( false );
    if (debug)
      {
	std::cout<<"checktime "<<bx<<" "<<timeDifference<<" "<<simhitTime<<" "<<referenceTime<<" "<<tof<<" "<<averagePropagationTime<<std::endl;
      }
  }
  return bx;
}

