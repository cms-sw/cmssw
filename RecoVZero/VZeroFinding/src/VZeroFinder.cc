#include "RecoVZero/VZeroFinding/interface/VZeroFinder.h"

#include "DataFormats/Math/interface/Vector3D.h"

using namespace std;

/*****************************************************************************/
VZeroFinder::VZeroFinder
  (const edm::EventSetup& es,
   const edm::ParameterSet& pset)
{
  // Get track-pair level cuts
  maxDcaR = pset.getParameter<double>("maxDcaR");
  maxDcaZ = pset.getParameter<double>("maxDcaZ");

  // Get mother cuts
  minCrossingRadius = pset.getParameter<double>("minCrossingRadius");
  maxCrossingRadius = pset.getParameter<double>("maxCrossingRadius");
  maxImpactMother   = pset.getParameter<double>("maxImpactMother");

  // Get magnetic field
  edm::ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);
  theMagField = magField.product();

  // Get closest approach finder
  theApproach = new ClosestApproachInRPhi();
}

/*****************************************************************************/
VZeroFinder::~VZeroFinder()
{
  delete theApproach;
}

/*****************************************************************************/
FreeTrajectoryState VZeroFinder::getTrajectory(const reco::Track& track)
{ 
  GlobalPoint position(track.vertex().x(),
                       track.vertex().y(),
                       track.vertex().z());

  GlobalVector momentum(track.momentum().x(),
                        track.momentum().y(),
                        track.momentum().z());

  GlobalTrajectoryParameters gtp(position,momentum,
                                 track.charge(),theMagField);
  
  FreeTrajectoryState fts(gtp);

  return fts; 
} 

/*****************************************************************************/
bool VZeroFinder::checkTrackPair(const reco::Track& posTrack,
                                 const reco::Track& negTrack,
                                 const reco::VertexCollection* vertices,
                                       reco::VZeroData& data)
{
  // Get trajectories
  FreeTrajectoryState posTraj = getTrajectory(posTrack);
  FreeTrajectoryState negTraj = getTrajectory(negTrack);

  // Closest approach
  bool status = theApproach->calculate(posTraj,negTraj);
  pair<GlobalTrajectoryParameters, GlobalTrajectoryParameters>  
    gtp = theApproach->trajectoryParameters();

  // Closest points
  pair<GlobalPoint, GlobalPoint>
    points(gtp.first.position(), gtp.second.position());

  // Momenta at closest point
  pair<GlobalVector,GlobalVector>
    momenta(gtp.first.momentum(), gtp.second.momentum());

  // dcaR
  float dcaR = (points.first - points.second).perp();
  GlobalPoint crossing(0.5*(points.first.x() + points.second.x()),
                       0.5*(points.first.y() + points.second.y()),
                       0.5*(points.first.z() + points.second.z()));

  if(dcaR < maxDcaR &&
     crossing.perp() > minCrossingRadius &&
     crossing.perp() < maxCrossingRadius)
  {
    // dcaZ
    float theta = 0.5*(posTrack.theta() + negTrack.theta());
    float dcaZ = fabs((points.first - points.second).z()) * sin(theta); 

    if(dcaZ < maxDcaZ)
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

      if(impact < maxImpactMother)
      {
        // Armenteros variables
        float pt =
          (momenta.first.cross(momenta.second)).mag()/momentum.mag();
        float alpha =
          (momenta.first.mag2() - momenta.second.mag2())/momentum.mag2();

        // Fill data
        data.dcaR = dcaR;
        data.dcaZ = dcaZ;
        data.crossingPoint = crossing;
        data.impactMother  = impact;
        data.armenterosPt    = pt;
        data.armenterosAlpha = alpha;

        return true;
      }
    }
  }

  return false;
}

