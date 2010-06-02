#include "RecoVZero/VZeroFinding/interface/VZeroFinder.h"

#include "DataFormats/Math/interface/Vector3D.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

#include <utility>
using namespace std;

/*****************************************************************************/
VZeroFinder::VZeroFinder
  (const edm::EventSetup& es,
   const edm::ParameterSet& pset)
{
  // Get track-pair level cuts
  maxDca = pset.getParameter<double>("maxDca");

  // Get mother cuts
  minCrossingRadius = pset.getParameter<double>("minCrossingRadius");
  maxCrossingRadius = pset.getParameter<double>("maxCrossingRadius");
  maxImpactMother   = pset.getParameter<double>("maxImpactMother");

  // Get magnetic field
  edm::ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);
  theMagField = magField.product();
}

/*****************************************************************************/
VZeroFinder::~VZeroFinder()
{
}

/*****************************************************************************/
GlobalTrajectoryParameters VZeroFinder::getGlobalTrajectoryParameters
  (const reco::Track& track)
{
  GlobalPoint position(track.vertex().x(),
                       track.vertex().y(),
                       track.vertex().z());

  GlobalVector momentum(track.momentum().x(),
                        track.momentum().y(),
                        track.momentum().z());

  GlobalTrajectoryParameters gtp(position,momentum,
                                 track.charge(),theMagField);

  return gtp;
}

/*****************************************************************************/
GlobalVector VZeroFinder::rotate(const GlobalVector & p, double a)
{
  double pt = p.perp();
  return GlobalVector( -pt*cos(a), -pt*sin(a), p.z());
}

/*****************************************************************************/
bool VZeroFinder::checkTrackPair(const reco::Track& posTrack,
                                 const reco::Track& negTrack,
                                 const reco::VertexCollection* vertices,
                                       reco::VZeroData& data)
{
/*
  LogTrace("MinBiasTracking") << " [VZeroFinder] tracks"
   << " +" << posTrack.algoName() << " " << posTrack.d0()
   << " -" << negTrack.algoName() << " " << negTrack.d0();
*/

  // Get trajectories
  GlobalTrajectoryParameters posGtp = getGlobalTrajectoryParameters(posTrack);
  GlobalTrajectoryParameters negGtp = getGlobalTrajectoryParameters(negTrack);

  // Two track minimum distance
  TwoTrackMinimumDistance theMinimum(TwoTrackMinimumDistance::SlowMode);

  // Closest approach
  theMinimum.calculate(posGtp,negGtp);

  // Closest points
  pair<GlobalPoint, GlobalPoint> points = theMinimum.points();

  // Momenta at closest point
  pair<GlobalVector,GlobalVector> momenta;
  momenta.first  = rotate(posGtp.momentum(), theMinimum.firstAngle() );
  momenta.second = rotate(negGtp.momentum(), theMinimum.secondAngle());

  // dcaR
  float dca            = theMinimum.distance();
  GlobalPoint crossing = theMinimum.crossingPoint();

/*
  LogTrace("MinBiasTracking") << fixed << setprecision(2)
    << "  [VZeroFinder] dca    = "<<dca<<" (<"<<maxDca<<")";
  LogTrace("MinBiasTracking") << fixed << setprecision(2)
    << "  [VZeroFinder] crossR = "<<crossing.perp()<<" ("
    <<minCrossingRadius<<"< <"<<maxCrossingRadius<<")";
*/

  if(dca < maxDca &&
     crossing.perp() > minCrossingRadius &&
     crossing.perp() < maxCrossingRadius)
  {
    // Momentum of the mother
    GlobalVector momentum = momenta.first + momenta.second;
    float impact = -1.;

    if(vertices->size() > 0)
    {
      // Impact parameter of the mother wrt vertices, choose smallest
      for(reco::VertexCollection::const_iterator
          vertex = vertices->begin(); vertex!= vertices->end(); vertex++)
      {
        GlobalVector r(crossing.x(),
                       crossing.y(),
                       crossing.z() - vertex->position().z());
        GlobalVector p(momentum.x(),momentum.y(),momentum.z());
  
        GlobalVector b = r - (r*p)*p / p.mag2();
  
        float im = b.mag();
        if(im < impact || vertex == vertices->begin())
          impact = im; 
      }
    }
    else
    {
      // Impact parameter of the mother in the plane
      GlobalVector r_(crossing.x(),crossing.y(),0);
      GlobalVector p_(momentum.x(),momentum.y(),0);

      GlobalVector b_ = r_ - (r_*p_)*p_ / p_.mag2();
      impact = b_.mag(); 
    }

/*
    LogTrace("MinBiasTracking") << fixed << setprecision(2)
      << "  [VZeroFinder] impact = "<<impact<<" (<"<<maxImpactMother<<")";
*/

    if(impact < maxImpactMother)
    {
      // Armenteros variables
      float pt =
        (momenta.first.cross(momenta.second)).mag()/momentum.mag();
      float alpha =
        (momenta.first.mag2() - momenta.second.mag2())/momentum.mag2();

      // Fill data
      data.dca             = dca;
      data.crossingPoint   = crossing;
      data.impactMother    = impact;
      data.armenterosPt    = pt;
      data.armenterosAlpha = alpha;
      data.momenta         = momenta;

/*
      LogTrace("MinBiasTracking") << fixed << setprecision(2)
      << "  [VZeroFinder] success {alpha = "<<alpha<<", pt = "<<pt<<"}";
*/

      return true;
    }
  }

  return false;
}

