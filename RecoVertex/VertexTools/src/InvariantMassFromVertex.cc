#include "RecoVertex/VertexTools/interface/InvariantMassFromVertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


InvariantMassFromVertex::LorentzVector InvariantMassFromVertex::p4 (const CachingVertex<5>& vertex,
                          const double mass) const
{
  return p4(vertex, vector<double>(vertex.tracks().size(), mass));
}

InvariantMassFromVertex::LorentzVector InvariantMassFromVertex::p4 (const CachingVertex<5>& vertex,
                          const vector<double> & masses) const
{

  LorentzVector totalP4;

  // Check that tkToTkCovarianceIsAvailable
  if (!vertex.tkToTkCovarianceIsAvailable()) {
    LogDebug("InvariantMassFromVertex")
	<< "Fit failed: vertex has not been smoothed\n";
    return totalP4;
  }

  if (vertex.tracks().size() != masses.size()) {
    LogDebug("InvariantMassFromVertex")
	<< "Vector of masses does not have the same size as tracks in vertex\n";
    return totalP4;
  }


  vector<RefCountedVertexTrack> refTracks = vertex.tracks();
  vector<RefCountedVertexTrack>::const_iterator i_s = refTracks.begin();
  vector<double>::const_iterator i_m = masses.begin();

  for( ;i_s !=refTracks.end(), i_m != masses.end(); ++i_s, ++i_m) {
    GlobalVector momentum = (**i_s).refittedState()->freeTrajectoryState().momentum();
    totalP4 += LorentzVector(momentum.x(), momentum.y(), momentum.z(), *i_m);
  } 
  return totalP4;
}

GlobalVector InvariantMassFromVertex::momentum(const CachingVertex<5>& vertex) const
{
 GlobalVector momentum_;

  // Check that tkToTkCovarianceIsAvailable
  if (!vertex.tkToTkCovarianceIsAvailable()) {
    LogDebug("InvariantMassFromVertex")
	<< "Fit failed: vertex has not been smoothed\n";
   return momentum_;
  }

  vector<RefCountedVertexTrack> refTracks = vertex.tracks();
  vector<RefCountedVertexTrack>::const_iterator i_s = refTracks.begin();

  for( ;i_s !=refTracks.end() ; ++i_s) {
    momentum_ += (**i_s).refittedState()->freeTrajectoryState().momentum();
  } 
  return momentum_;

}


Measurement1D InvariantMassFromVertex::invariantMass(const CachingVertex<5>& vertex,
                          const double mass) const
{
  return invariantMass(vertex, vector<double>(vertex.tracks().size(), mass));
}


Measurement1D InvariantMassFromVertex::invariantMass(const CachingVertex<5>& vertex,
                          const vector<double> & masses) const
{

  // Check that tkToTkCovarianceIsAvailable
  if (!vertex.tkToTkCovarianceIsAvailable()) {
    LogDebug("InvariantMassFromVertex")
	<< "Fit failed: vertex has not been smoothed\n";
    return Measurement1D(0.,0.);
  }
  if (vertex.tracks().size() != masses.size()) {
    LogDebug("InvariantMassFromVertex")
	<< "Vector of masses does not have the same size as tracks in vertex\n";
    return Measurement1D(0.,0.);
  }

  LorentzVector totalP4 = p4(vertex, masses);
  double u = uncertainty(totalP4, vertex, masses);
  cout << u<<endl;
  return Measurement1D(totalP4.M(), u );

}

//method returning the full covariance matrix
//of new born kinematic particle
double InvariantMassFromVertex::uncertainty(const LorentzVector & totalP4, 
	const CachingVertex<5>& vertex, const vector<double> & masses) const
{
  vector<RefCountedVertexTrack> refTracks = vertex.tracks();
  int size = refTracks.size();
  AlgebraicMatrix cov(3*size,3*size);
  AlgebraicMatrix jac(1,3*size);

  double energy_total = totalP4.E();

  vector<RefCountedVertexTrack>::const_iterator rt_i = refTracks.begin();
  vector<double>::const_iterator i_m = masses.begin();

  int i_int = 0;
  for( ;rt_i !=refTracks.end(), i_m != masses.end(); ++rt_i, ++i_m) {

    double a;
    AlgebraicVector5 param = (**rt_i).refittedState()->parameters(); // rho, theta, phi,tr_im, z_im
    double rho = param[0];
    double theta = param[1];
    double phi = param[2];
    double mass = *i_m;

    if ((**rt_i).linearizedTrack()->charge()!=0) {
      a = -(**rt_i).refittedState()->freeTrajectoryState().parameters().magneticFieldInInverseGeV(vertex.position()).z()
      		* (**rt_i).refittedState()->freeTrajectoryState().parameters ().charge();
      if (a==0.) throw cms::Exception("InvariantMassFromVertex", "Field is 0");
    } else {
      a = 1;
    }

    double energy_local  = sqrt(a*a/(rho*rho)*(1+1/(tan(theta)*tan(theta)))  + mass*mass);

    jac(1,i_int*3+1) = (-(energy_total/energy_local*a*a/(rho*rho*rho*sin(theta)*sin(theta)) )
  		  + totalP4.X()*a/(rho*rho)*cos(phi) + totalP4.Y()*a/(rho*rho)*sin(phi)
		  + totalP4.Z()*a/(rho*rho*tan(theta)) )/totalP4.M();	//dm / drho

    jac(1,i_int*3+2) = (-(energy_total/energy_local*a*a/(rho*rho*sin(theta)*sin(theta)*tan(theta)) )
		  + totalP4.Z()*a/(rho*sin(theta)*sin(theta)) )/totalP4.M();//dm d theta

    jac(1,i_int*3+3) = ( totalP4.X()*sin(phi) - totalP4.Y()*cos(phi) )*a/(rho*totalP4.M());	//dm/dphi

    // momentum corellatons: diagonal elements of the matrix
    cov.sub(i_int*3 + 1, i_int*3 + 1,asHepMatrix<6>((**rt_i).fullCovariance()).sub(4,6));

    //off diagonal elements: track momentum - track momentum corellations

    int j_int = 0;
    for(vector<RefCountedVertexTrack>::const_iterator rt_j = refTracks.begin(); rt_j != refTracks.end(); rt_j++) {
      if(i_int < j_int) {
	AlgebraicMatrix i_k_cov_m = asHepMatrix<3,3>(vertex.tkToTkCovariance((*rt_i),(*rt_j)));
	cov.sub(i_int*3 + 1, j_int*3 + 1,i_k_cov_m);
	cov.sub(j_int*3 + 1, i_int*3 + 1,i_k_cov_m.T());
      }
      j_int++;
    }
    i_int++;
  }
//   cout<<"jac"<<jac<<endl;
//   cout<<"cov"<<cov<<endl;
//   cout << "final result"<<(jac*cov*jac.T())<<endl;

  return sqrt((jac*cov*jac.T())(1,1));
}
