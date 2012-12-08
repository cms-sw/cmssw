#include "SimMuon/GEMDigitizer/src/GEMSynchronizer.h"
#include "SimMuon/GEMDigitizer/src/GEMSimSetUp.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "CLHEP/Random/RandGaussQ.h"


using namespace std;

namespace 
{
  // "magic" parameter for cosmics
  const double COSMIC_PAR =  37.62;
}


GEMSynchronizer::GEMSynchronizer(const edm::ParameterSet& config):
  gauss1_(0), gauss2_(0)
{
  timeRes_ = config.getParameter<double>("timeResolution");
  timOff_ = config.getParameter<double>("timingOffset");
  dtimCs_ = config.getParameter<double>("deltatimeAdjacentStrip");
  resEle_ = config.getParameter<double>("timeJitter");
  sspeed_ = config.getParameter<double>("signalPropagationSpeed");
  lbGate_ = config.getParameter<double>("linkGateWidth");
  cosmics_ = config.getParameter<bool>("cosmics");

  const double c = 299792458;// [m/s]

  //light speed in [cm/ns]
  const double cspeed = c * 1e+2 * 1e-9;

  //signal propagation speed [cm/ns]
  sspeed_ = sspeed_ * cspeed;
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
  float timeref = simsetup->getTime(simhit->detUnitId());

  int bx = -999;
  LocalPoint simHitPos = simhit->localPosition();
  float tof = simhit->timeOfFlight();
  
  //automatic variable to prevent memory leak
  
  float rr_el = gauss1_->fire(0., resEle_);
  
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
    float half_strip_length = 0.;

    if(shRoll->id().region() == 0)
    {
      throw cms::Exception("Geometry")
        << "Exception coming from GEMSynchronizer - this GEM id is from barrel, which cannot happen:  "<<shRoll->id()<< "\n";
    }
    else
    {
      const TrapezoidalStripTopology* top= dynamic_cast<const TrapezoidalStripTopology*> (&(shRoll->topology()));
      half_strip_length = 0.5 * top->stripLength();
      distanceFromEdge = half_strip_length - simHitPos.y();
    }

    float prop_time =  distanceFromEdge/sspeed_;

    double rr_tim1 = gauss2_->fire(0., resEle_);
    double total_time = tof + prop_time + timOff_ + rr_tim1 + rr_el;
    
    // Bunch crossing assignment
    double time_differ = 0.;

    if(cosmics_)
    {
      time_differ = (total_time - (timeref + ( half_strip_length/sspeed_ + timOff_)))/COSMIC_PAR;
    }
    else
    {
      time_differ = total_time - (timeref + ( half_strip_length/sspeed_ + timOff_));
    }
     
    double inf_time = 0;
    double sup_time = 0;

    for(int n = -5; n <= 5; ++n)
    {
      if(cosmics_)
      {
        inf_time = (-lbGate_/2 + n*lbGate_ )/COSMIC_PAR;
        sup_time = ( lbGate_/2 + n*lbGate_ )/COSMIC_PAR;
      }
      else
      {
        inf_time = -lbGate_/2 + n*lbGate_;
        sup_time =  lbGate_/2 + n*lbGate_;
      }

      if(inf_time < time_differ && time_differ < sup_time)
      {
        bx = n;
        break;
      }
    }
  }

  return bx;
}

