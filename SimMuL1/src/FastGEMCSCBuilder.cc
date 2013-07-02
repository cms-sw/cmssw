//

#include "GEMCode/SimMuL1/interface/FastGEMCSCBuilder.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include "GEMCode/GEMValidation/src/SimTrackMatchManager.h"

using namespace std;
using namespace matching;


FastGEMCSCBuilder::FastGEMCSCBuilder(const edm::ParameterSet& ps, CLHEP::HepRandomEngine& eng)
: zOddGEM_(ps.getParameter<vector<double> >("zOddGEM"))
, zEvenGEM_(ps.getParameter<vector<double> >("zEvenGEM"))
, phiSmearCSC_(ps.getParameter<vector<double> >("phiSmearCSC"))
, phiSmearGEM_(ps.getParameter<vector<double> >("phiSmearGEM"))
, fitterXZ_(new TLinearFitter(1, "pol1"))
, fitterYZ_(new TLinearFitter(1, "pol1"))
, flat_(new CLHEP::RandFlat(eng))
{
  fitterXZ_->StoreData(1);
  fitterYZ_->StoreData(1);

  // these configuration vectors have to have 1+10 elements corresponding to 10 chamber types
  assert(zOddGEM_.size() == 11);
  assert(zEvenGEM_.size() == 11);
  assert(phiSmearCSC_.size() == 11);
  assert(phiSmearCSC_.size() == 11);
  // Sanity check: negative parameters values would mean chambers are not supposed to be used
  // non-used chamber types have to be the same between all the vectors
  for (int i=1; i<11; ++i)
  {
    if (zOddGEM_[i] < 0.)
    {
      assert(zEvenGEM_[i] < 0. && phiSmearCSC_[i] < 0. && phiSmearCSC_[i] < 0.);
    }
  }
}


std::vector<unsigned int> FastGEMCSCBuilder::getChamberIds()
{
  std::vector<unsigned int> result;
  for (auto &p: stubs_map_)
  {
    result.push_back(p.first);
  }
  return result;
}


void FastGEMCSCBuilder::build(const SimHitMatcher& match_sh)
{
  // start with clean plate
  stubs_map_.clear();

  const SimTrack &t = match_sh.trk();

  // retrieve all types of simhits from SimHitMatcher
  auto csc_ch_ids = match_sh.chamberIdsCSC(0);
  for(auto d: csc_ch_ids)
  {
    CSCDetId id(d);

    int nlayers = match_sh.nLayersWithHitsInSuperChamber(d);
    if (nlayers < 4) continue;

    int ch_type = id.iChamberType();
    bool odd = id.chamber() & 1;

    // --- determine the z-position of gem
    double z_gem;
    if (odd) {
      if (id.endcap() == 1) z_gem =  zOddGEM_[ch_type];
      else                  z_gem = -zOddGEM_[ch_type];
    }
    else {
      if (id.endcap() == 1) z_gem =  zEvenGEM_[ch_type];
      else                  z_gem = -zEvenGEM_[ch_type];
    }

    SimStub stub(z_gem);

    // Linear model of stub constructed from muon simhits
    //
    // Symmetric form of line equation: (x - x0)/a = (y - y0)/b = (z - z0)/c
    // Only 4 of 6 parameters are independent, so with no loss of generality
    // we can set c = 1, and, e.g.,  z0 = zkey, where for zkey we would use position of chamber's key layer
    // We'll do two linear fits for x and y dependency on z (sine z's are fixed by detector positions):
    // x(z) = x0 + a *(z - z0) = xz0 + xz1*z, where xz0 = x0 - xz1*z0, xz1 = a
    // y(z) = y0 + b *(z - z0) = yz0 + yz1*z, where yz0 = y0 - yz1*z0, yz1 = b
    // we find xz0, xz1, yz0, yz1 from linear fits

    //cout<<" hitXZ ";
    const auto& hits = match_sh.hitsInChamber(d);
    for (auto& h: hits)
    {
      stub.addHalfStrips( match_sh.hitStripsInDetId(h.detUnitId(), 1) ); // use single HS margin
      stub.addWireGroups( match_sh.hitWiregroupsInDetId(h.detUnitId(), 1) ); // use single WG margin

      GlobalPoint gp = csc_geo_->idToDet(h.detUnitId())->surface().toGlobal(h.entryPoint());
      //LocalPoint lp = csc_geo_->idToDet(id.chamberId())->surface().toLocal(gp);
      //cout<< lp.x() <<" "<<gp.z()<<"  ";

      double z[1] = {gp.z()};
      fitterXZ_->AddPoint(z, gp.x()); // x(z)
      fitterYZ_->AddPoint(z, gp.y()); // y(z)
    }
    //cout<<endl;
    fitterXZ_->Eval();
    fitterYZ_->Eval();

    stub.setFitParameters(
        fitterXZ_->GetParameter(0), fitterXZ_->GetParameter(1),
        fitterYZ_->GetParameter(0), fitterYZ_->GetParameter(1) );

    //double xz0e = fitterXZ_->GetParError(0);
    //double xz1e = fitterXZ_->GetParError(1);
    //double yz0e = fitterYZ_->GetParError(0);
    //double yz1e = fitterYZ_->GetParError(1);

    // clean-up the fitters
    fitterXZ_->ClearPoints();
    fitterYZ_->ClearPoints();

    // --- find stub global position at CSC chamber key layer
    CSCDetId key_id(id.endcap(), id.station(), id.ring(), id.chamber(), CSCConstants::KEY_CLCT_LAYER);
    GlobalPoint gp_key = csc_geo_->idToDet(key_id)->surface().toGlobal(LocalPoint(0.,0.,0.));
    // fitted SimHits stub position at key layer
    GlobalPoint gp_csc = stub.globalPointAtZ( gp_key.z() );
    // optionally, do phi smearing
    if (phiSmearCSC_[ch_type] > 0.)
    {
      auto theta = gp_csc.theta(); // Geom::Theta<> object
      auto phi   = gp_csc.phi(); // Geom::Phi<> object
      auto r     = gp_csc.mag();
      float smear = flat_->fire(phiSmearCSC_[ch_type]) - phiSmearCSC_[ch_type] * 0.5;
      phi += smear;
      gp_csc = GlobalPoint(GlobalPoint::Spherical(theta, phi, r));
    }
    stub.setCSC(gp_csc);

    GlobalPoint gp_gem_lin = stub.globalPointAtZ( z_gem );
    float gem_phi_smear = 0.;
    if (phiSmearGEM_[ch_type] > 0.)
    {
      auto theta = gp_gem_lin.theta(); // Geom::Theta<> object
      auto phi   = gp_gem_lin.phi(); // Geom::Phi<> object
      auto r     = gp_gem_lin.mag();
      gem_phi_smear = flat_->fire(phiSmearGEM_[ch_type]) - phiSmearGEM_[ch_type] * 0.5;
      phi += gem_phi_smear;
      gp_gem_lin = GlobalPoint(GlobalPoint::Spherical(theta, phi, r));
    }
    stub.setGEMLinear( gp_gem_lin );

    // propagate to z_gem
    GlobalVector inner_vector = match_sh.trk().momentum().P() * stub.direction();
    GlobalPoint gp_gem_prop = match_sh.propagateToZ(gp_csc, inner_vector, z_gem);
    if (phiSmearGEM_[ch_type] > 0.)
    {
      auto theta = gp_gem_prop.theta(); // Geom::Theta<> object
      auto phi   = gp_gem_prop.phi(); // Geom::Phi<> object
      auto r     = gp_gem_prop.mag();
      phi += gem_phi_smear;  // use the same smear amount as for the linear case
      gp_gem_prop = GlobalPoint(GlobalPoint::Spherical(theta, phi, r));
    }
    stub.setGEMPropagator( gp_gem_prop );

    // debug printout
    if (1)
    {
      double dphi_lin = stub.dPhiGEMCSCLinear();
      double dphi_prop = stub.dPhiGEMCSCPropagator();
      cout<<" gp_sh"<<ch_type<<((t.charge()>0)? '+' : '-' )<<" "<<t.momentum().eta()<<" "<<t.momentum().pt()<<" "<<t.charge()<<" "
        <<odd<<" "<<id.chamber()<<" "<<gp_csc<<" "<<gp_gem_lin<<" "<<gp_gem_prop<<"  "
        << dphi_lin <<" "<< dphi_prop <<" "<< dphi_lin - dphi_prop << endl;
    }

    if ( stub.isValid() )
    {
      if (stubs_map_.find(d) == stubs_map_.end())
      {
        stubs_map_[d] = vector<SimStub>();
      }
      stubs_map_[d].push_back(stub);
    }
    else
    {
      cout<<"Error: non-valid SimStub: "<< stub <<endl;
    }
  }

}


// ------------ SimStub implementation  ------------


SimStub::SimStub(double z_gem)
: z_gem_(z_gem)
, min_hs_(999)
, max_hs_(-1)
, min_wg_(999)
, max_wg_(-1)
{}


void SimStub::setFitParameters(double x0, double x1, double y0, double y1)
{
  x0_ = x0;
  x1_ = x1;
  y0_ = y0;
  y1_ = y1;

  GlobalVector direction(x1, y1, 1.);
  // normalize direction_ to 1.
  direction_ = direction / direction.mag();
}


void SimStub::addHalfStrips(std::set<int> hs)
{
  int hs_min = *hs.begin();
  int hs_max = *hs.rbegin();
  if (hs_min < min_hs_) min_hs_ = hs_min;
  if (hs_max > max_hs_) max_hs_ = hs_max;
}


void SimStub::addWireGroups(std::set<int> wg)
{
  int wg_min = *wg.begin();
  int wg_max = *wg.rbegin();
  if (wg_min < min_wg_) min_wg_ = wg_min;
  if (wg_max > max_wg_) max_wg_ = wg_max;
}

std::ostream & operator<<(std::ostream & o, const SimStub& s)
{
  o << " csc "<<s.gp_csc_<<"  gem_lin "<<s.gp_gem_lin_<<"  gem_prop "<<s.gp_gem_prop_
    << "  dphi_lin "<<s.dPhiGEMCSCLinear()<<"  dphi_prop "<<s.dPhiGEMCSCPropagator()
    << " hs: ["<<s.min_hs_<<","<<s.max_hs_<<"]  wg: ["<<s.min_wg_<<","<<s.max_wg_<<"] ";
  return o;
}
