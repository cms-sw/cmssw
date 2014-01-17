#include "GEMCode/SimMuL1/interface/MuGeometryHelpers.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include <utility>
using namespace std;
using std::cout;
using std::endl;
using namespace mugeo;

const float mugeo::MuGeometryAreas::csc_ch_radius[CSC_TYPES+1] = {0., 128., 203.25, 369.75, 594.1, 239.05, 525.55, 251.75, 525.55, 261.7, 525.55};
const float mugeo::MuGeometryAreas::csc_ch_halfheight[CSC_TYPES+1] = {0., 22., 53.25, 87.25, 82.1, 94.85, 161.55, 84.85, 161.55, 74.7, 161.55};
const float mugeo::MuGeometryAreas::dt_ch_z[DT_TYPES+1] = {0., 58.7, 273, 528, 58.7, 273, 528, 58.7, 273, 528, 58.7, 273, 528};
const float mugeo::MuGeometryAreas::dt_ch_halfspanz[DT_TYPES+1] = {0., 58.7, 117.4, 117.4, 58.7, 117.4, 117.4, 58.7, 117.4, 117.4, 58.7, 117.4, 117.4};


// ================================================================================================
void mugeo::MuGeometryAreas::calculateCSCDetectorAreas(const CSCGeometry* g)
{
  for (int i=0; i<=CSC_TYPES; i++) csc_total_areas_cm2[i]=0.;

  for(std::vector<CSCLayer*>::const_iterator it = g->layers().begin(); it != g->layers().end(); it++)
  {
    if( dynamic_cast<CSCLayer*>( *it ) == 0 ) continue;

    CSCLayer* layer = dynamic_cast<CSCLayer*>( *it );
    CSCDetId id = layer->id();
    int ctype = id.iChamberType();

    const CSCWireTopology*  wire_topo  = layer->geometry()->wireTopology();
    const CSCStripTopology* strip_topo = layer->geometry()->topology();
    float b = wire_topo->narrowWidthOfPlane();
    float t = wire_topo->wideWidthOfPlane();
    float w_h   = wire_topo->lengthOfPlane();
    float s_h = fabs(strip_topo->yLimitsOfStripPlane().first - strip_topo->yLimitsOfStripPlane().second);
    float h = (w_h < s_h)? w_h : s_h;

    // special cases:
    if (ctype==1) // ME1/1a
    {
      h += -0.5; // adjustment in order to agree with the official me1a height number
      t = ( b*(w_h - h) + t*h )/w_h;
    }
    if (ctype==2) // ME1/1a
    {
      h += -1.;
      b = ( b*h + t*(w_h - h) )/w_h;
    }

    float layer_area = h*(t + b)*0.5;
    csc_total_areas_cm2[0] += layer_area;
    csc_total_areas_cm2[ctype] += layer_area;

    if (id.layer()==1) cout<<"CSC type "<<ctype<<"  "<<id<<"  layer area: "<<layer_area<<" cm2   "
        <<"  b="<<b<<" t="<<t<<" h="<<h<<"  w_h="<<w_h<<" s_h="<<s_h<<endl;
  }
  cout<<"========================"<<endl;
  cout<<"= CSC chamber sensitive areas per layer (cm2):"<<endl;
  for (int i=1; i<=CSC_TYPES; i++) cout<<"= "<<csc_type[i]<<" "<<csc_total_areas_cm2[i]/6./2./csc_radial_segm[i]<<endl;
  cout<<"========================"<<endl;
}


// ================================================================================================
void mugeo::MuGeometryAreas::calculateGEMDetectorAreas(const GEMGeometry* g)
{
  std::vector<float> emptyv(12, 0.);
  float minr[GEM_TYPES+1], maxr[GEM_TYPES+1];
  for (int i=0; i<=GEM_TYPES; i++)
  {
    gem_total_areas_cm2[i]=0.;
    gem_total_part_areas_cm2[i] = emptyv;
    gem_part_radius[i] = emptyv;
    gem_part_halfheight[i] = emptyv;
    minr[i] = 9999.;
    maxr[i] = 0.;
  }


  auto etaPartitions = g->etaPartitions();
  for(auto p: etaPartitions)
  {
    GEMDetId id = p->id();
    int t = type(id);
    int part = id.roll();

    const TrapezoidalStripTopology* top = dynamic_cast<const TrapezoidalStripTopology*>(&(p->topology()));
    float xmin = top->localPosition(0.).x();
    float xmax = top->localPosition((float)p->nstrips()).x();
    float rollarea = top->stripLength() * (xmax - xmin);
    gem_total_areas_cm2[0] += rollarea;
    gem_total_areas_cm2[t] += rollarea;
    gem_total_part_areas_cm2[0][0] += rollarea;
    gem_total_part_areas_cm2[t][part] += rollarea;
    cout<<"Partition: "<<id.rawId()<<" "<<id<<" area: "<<rollarea<<" cm2"<<endl;

    GlobalPoint gp = g->idToDet(id)->surface().toGlobal(LocalPoint(0.,0.,0.));
    gem_part_radius[t][part] = gp.perp();
    gem_part_halfheight[t][part] = top->stripLength()/2.;

    if (maxr[t] < gp.perp() + top->stripLength()/2.) maxr[t] = gp.perp() + top->stripLength()/2.;
    if (minr[t] > gp.perp() - top->stripLength()/2.) minr[t] = gp.perp() - top->stripLength()/2.;
  }

  for (int t=1; t<=GEM_TYPES; t++)
  {
    gem_part_radius[t][0] = (minr[t] + maxr[t])/2.;
  }

  cout<<"========================"<<endl;
  cout<<"= GEM chamber total sensitive areas (cm2):"<<endl;
  for (int i=0; i<=GEM_TYPES; i++) cout<<"= "<<gem_type[i]<<" "<<gem_total_areas_cm2[i]<<endl;
  cout<<"= GEM chamber sensitive areas per layer (cm2):"<<endl;
  for (int i=0; i<=GEM_TYPES; i++) cout<<"= "<<gem_type[i]<<" "<<gem_total_areas_cm2[i]/2./2./gem_radial_segm[i]<<endl;
  cout<<"========================"<<endl;
}


// ================================================================================================
void mugeo::MuGeometryAreas::calculateDTDetectorAreas(const DTGeometry* g)
{
  for (int i=0; i<=DT_TYPES; i++) dt_total_areas_cm2[i]=0.;

  for(std::vector<DTLayer*>::const_iterator it = g->layers().begin(); it != g->layers().end(); it++)
  {
    if( dynamic_cast<DTLayer*>( *it ) == 0 ) continue;

    DTLayer* layer = dynamic_cast<DTLayer*>( *it );
    DTWireId id = (DTWireId) layer->id();
    int t = type(id);

    const DTTopology& topo = layer->specificTopology();
    // cell's sensible width * # cells
    float w = topo.sensibleWidth() * topo.channels();
    float l = topo.cellLenght();

    float layer_area = w*l;
    dt_total_areas_cm2[0] += layer_area;
    dt_total_areas_cm2[t] += layer_area;

    if (id.layer()==1) cout<<"DT type "<<t<<"  "<<id<<"  layer area: "<<layer_area<<" cm2   "
        <<"  w="<<w<<" l="<<l<<" ncells="<<topo.channels()<<endl;
  }

  cout<<"========================"<<endl;
  cout<<"= DT *total* sensitive areas (cm2):"<<endl;
  for (int i=0; i<=DT_TYPES; i++) cout<<"= "<<dt_type[i]<<" "<<dt_total_areas_cm2[i]<<endl;
  cout<<"========================"<<endl;
}


// ================================================================================================
void mugeo::MuGeometryAreas::calculateRPCDetectorAreas(const RPCGeometry* g)
{
  for (int i=0; i<=RPCB_TYPES; i++) rpcb_total_areas_cm2[i]=0.;
  for (int i=0; i<=RPCF_TYPES; i++) rpcf_total_areas_cm2[i]=0.;

  // adapted from Piet's RPCGeomAnalyzer

  for(std::vector<RPCRoll*>::const_iterator it = g->rolls().begin(); it != g->rolls().end(); it++)
  {
    if( dynamic_cast<RPCRoll*>( *it ) != 0 ) { // check if dynamic cast is ok: cast ok => 1
      RPCRoll* roll = dynamic_cast<RPCRoll*>( *it );
      RPCDetId id = roll->id();
      //RPCGeomServ rpcsrv(detId);
      //std::string name = rpcsrv.name();
      if (id.region() == 0) {
        const RectangularStripTopology* top_ = dynamic_cast<const RectangularStripTopology*>(&(roll->topology()));
        float xmin = (top_->localPosition(0.)).x();
        float xmax = (top_->localPosition((float)roll->nstrips())).x();
        float rollarea = top_->stripLength() * (xmax - xmin);
        rpcb_total_areas_cm2[0] += rollarea;
        rpcb_total_areas_cm2[type(id)] += rollarea;
        // cout<<"Roll: RawId: "<<id.rawId()<<" Name: "<<name<<" RPCDetId: "<<id<<" rollarea: "<<rollarea<<" cm2"<<endl;
      }
      else
      {
        const TrapezoidalStripTopology* top_=dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology()));
        float xmin = (top_->localPosition(0.)).x();
        float xmax = (top_->localPosition((float)roll->nstrips())).x();
        float rollarea = top_->stripLength() * (xmax - xmin);
        rpcf_total_areas_cm2[0] += rollarea;
        rpcf_total_areas_cm2[type(id)] += rollarea;
        // cout<<"Roll: RawId: "<<id.rawId()<<" Name: "<<name<<" RPCDetId: "<<id<<" rollarea: "<<rollarea<<" cm2"<<endl;
      }
    }
  }
  cout<<"========================"<<endl;
  cout<<"= RPCb *total* sensitive areas (cm2):"<<endl;
  for (int i=0; i<=RPCB_TYPES; i++) cout<<"= "<<rpcb_type[i]<<" "<<rpcb_total_areas_cm2[i]<<endl;
  cout<<"= RPCf chamber sensitive areas (cm2):"<<endl;
  for (int i=0; i<=RPCF_TYPES; i++) cout<<"= "<<rpcf_type[i]<<" "<<rpcf_total_areas_cm2[i]/2./rpcf_radial_segm[i]<<endl;
  cout<<"========================"<<endl;
}



// ================================================================================================
void mugeo::MuFiducial::buildGEMLUT()
{
  auto etaPartitions = gemGeometry->etaPartitions();
  for(auto roll: etaPartitions)
  {
    GEMDetId rId(roll->id());
              
    const BoundPlane& bSurface(roll->surface());
    //    const StripTopology* topology(&(roll->specificTopology()));
              
    // base_bottom, base_top, height, strips, pads (all half length)
    auto& parameters(roll->specs()->parameters());
    float bottomEdge(parameters[0]);
    float topEdge(parameters[1]);
    float height(parameters[2]);
    //    float nStrips(parameters[3]);
              
    LocalPoint  l1Top(-topEdge, height, 0.);
    //    LocalPoint  l2Top(topEdge, height, 0.);
    GlobalPoint g1Top(bSurface.toGlobal(l1Top));
    //    GlobalPoint g2Top(bSurface.toGlobal(l2Top));
              
    //    LocalPoint  l1Bottom(-bottomEdge, -height, 0.);
    LocalPoint  l2Bottom(bottomEdge, -height, 0.);
    //    GlobalPoint g1Bottom(bSurface.toGlobal(l1Bottom));
    GlobalPoint g2Bottom(bSurface.toGlobal(l2Bottom));
              
    double t1R(g1Top.perp());
//     double t2R(g2Top.perp());
    double t1phi(static_cast<double>(g1Top.phi().degrees()));
//     double t2phi(static_cast<double>(std::round(g2Top.phi().degrees())));
    if (t1phi < 0) t1phi += 360;
//     if (t2phi < 0) t2phi += 360;
    
    double b2R(g2Bottom.perp());
//     double b2R(g2Bottom.perp());
    double b2phi(static_cast<double>(g2Bottom.phi().degrees()));
//     double b2phi(static_cast<double>(std::round(g2Bottom.phi().degrees())));
    if (b2phi < 0) b2phi += 360;
//     if (b2phi < 0) b2phi += 360;

    double phiMin;
    double phiMax;

    if (rId.chamber() % 2==0 && rId.region()==1){
      phiMin = b2phi; phiMax = t1phi;
    }
    else if (rId.chamber() % 2!=0 && rId.region()==-1){
      //phiMin = t1phi; phiMax = b2phi;
      phiMin = b2phi; phiMax = t1phi;
    }
    else if (rId.chamber() % 2!=0 && rId.region()==1){
      //      phiMin = b2phi; phiMax = t1phi;
      phiMin = t1phi; phiMax = b2phi;
    }
    else {
      //phiMin = b2phi; phiMax = t1phi;
      phiMin = t1phi; phiMax = b2phi;
    }

//     if (rId.chamber() % 2==0) std::cout << "even chamber " <<  rId << " "<<phiMin << " "<<phiMax<< std::endl;
//     else                      std::cout << "odd chamber " <<  rId << " "<<phiMin << " "<<phiMax << std::endl;

    
    std::vector<double> values;
    values.push_back(b2R);
    values.push_back(t1R);
    values.push_back(phiMin);
    values.push_back(phiMax);
    values.push_back(0.);
    values.push_back(0.);
    
    gemLUT_[rId.rawId()] = values;
  }
}

// ================================================================================================
void mugeo::MuFiducial::buildCSCLUT()
{
  auto chambers = cscGeometry->chambers();
  for(auto ch: chambers)
  {    
    CSCDetId rId(ch->id());
    const BoundPlane& bSurface(ch->surface());
    auto layer1 = ch->layer(1);

    //    CSCDetId id = layer1->id();
    //    int ctype = id.iChamberType();

    const CSCWireTopology*  wire_topo  = layer1->geometry()->wireTopology();
    const CSCStripTopology* strip_topo = layer1->geometry()->topology();
    float b = wire_topo->narrowWidthOfPlane();
    float t = wire_topo->wideWidthOfPlane();
    float w_h   = wire_topo->lengthOfPlane();
    float s_h = fabs(strip_topo->yLimitsOfStripPlane().first - strip_topo->yLimitsOfStripPlane().second);
    float h = (w_h < s_h)? w_h : s_h;

    LocalPoint  l1Top(-t/2., h/2., 0.);
    LocalPoint  l2Top(t/2., h/2., 0.);
    GlobalPoint g1Top(bSurface.toGlobal(l1Top));
    GlobalPoint g2Top(bSurface.toGlobal(l2Top));
              
    LocalPoint  l1Bottom(-b/.2, -h/2., 0.);
    LocalPoint  l2Bottom(b/.2, -h/2., 0.);
    GlobalPoint g1Bottom(bSurface.toGlobal(l1Bottom));
    GlobalPoint g2Bottom(bSurface.toGlobal(l2Bottom));
              
    double t1R(g1Top.perp());
//     double t2R(g2Top.perp());
    double t1phi(static_cast<double>(std::round(g1Top.phi().degrees())));
//     double t2phi(static_cast<double>(std::round(g2Top.phi().degrees())));
    if (t1phi < 0) t1phi += 360;
//     if (t2phi < 0) t2phi += 360;
    
    double b1R(g1Bottom.perp());
//     double b2R(g2Bottom.perp());
    double b1phi(static_cast<double>(std::round(g1Bottom.phi().degrees())));
//     double b2phi(static_cast<double>(std::round(g2Bottom.phi().degrees())));
    if (b1phi < 0) b1phi += 360;
//     if (b2phi < 0) b2phi += 360;
    
    std::vector<double> values;
    values.push_back(b1R);
    values.push_back(t1R);
    values.push_back(b1phi);
    values.push_back(t1phi);
    values.push_back(0.);
    values.push_back(0.);
    
    cscLUT_[rId.rawId()] = values;
  }
}

std::set<uint32_t> mugeo::MuFiducial::gemDetIds(math::XYZVectorD p)
{
  std::set<uint32_t> result;
  const double r(p.Rho());
  double phi(p.Phi()*180./TMath::Pi()); //.degrees()
  if (phi < 0) phi += 360.;
  //  std::cout << "r " << r << " phi " << phi << " z " << p.z() << std::endl;
  
  for (auto it=gemLUT_.begin(); it!=gemLUT_.end(); ++it){
    // check if R and phi are withing limits
    // z match
    if (p.z()*(GEMDetId(it->first).region())<0) continue;
    // r match
    if (not(it->second.at(0) <= r and r <= it->second.at(1))) continue;
    // phi match
    if (not(it->second.at(2) <= phi and phi <= it->second.at(3))) continue;
//      std::cout << "detId " << it->first << " " << it->second.at(0) <<  " "<<it->second.at(1)<<" " 
//                << it->second.at(2)<< " " << it->second.at(3) << " " << GEMDetId(it->first).region() << std::endl;
//      std::cout << "match!!!" << std::endl;
  }
  return result;
}
