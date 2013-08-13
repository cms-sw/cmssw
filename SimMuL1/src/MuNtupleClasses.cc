#include "GEMCode/SimMuL1/interface/MuNtupleClasses.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

// ================================================================================================
void
MyCSCDetId::init(CSCDetId &id)
{
  e = id.endcap();
  s = id.station();
  r = id.ring();
  c = id.chamber();
  l = id.layer();
  t = type(id);
}


// ================================================================================================
void
MyCSCSimHit::init(PSimHit &sh, const CSCGeometry* csc_g, const ParticleDataTable * pdt)
{
  LocalPoint hitLP = sh.localPosition();
  pdg = sh.particleType();
  m = 0.00051;

  ParticleData const *pdata = 0;
  HepPDT::ParticleID particleType(pdg);
  if (particleType.isValid()) pdata = pdt->particle(particleType);
  if (pdata)  m = pdata->mass();
  //  cout<<"  "<<pdg<<" "<<pdata->name()<<" "<<m;
  else cout<<" pdg not in PDT: "<< pdg<<endl;

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  p = sh.pabs();
  t = sh.tof();
  trid = sh.trackId();

  CSCDetId layerId(sh.detUnitId());
  const CSCLayer* csclayer = csc_g->layer(layerId);
  GlobalPoint hitGP = csclayer->toGlobal(hitLP);

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gx = hitGP.x();
  gy = hitGP.y();
  gz = hitGP.z();

  w = csclayer->geometry()->wireGroup(csclayer->geometry()->nearestWire(hitLP));
  s = csclayer->geometry()->nearestStrip(hitLP);
}


bool MyCSCSimHit::operator<(const MyCSCSimHit & rhs) const
{
  // first sort by wire group, then by strip, then by TOF
  if (w==rhs.w)
  {
    if (s==rhs.s) return t<rhs.t;
    else return s<rhs.s;
  }
  else return w<rhs.w;
}


// ================================================================================================
void
MyCSCCluster::init(std::vector<MyCSCSimHit> &shits)
{
  hits = shits;
  nh = hits.size();
  mint = 1000000000.;
  maxt = -1.;
  minw = mins = 1000;
  maxw = maxs = -1;
  meant = 0;
  sigmat = 0;
  if (nh==0) return;
  r = hits[0].r;
  eta = hits[0].eta;
  phi = hits[0].phi;
  gx = hits[0].gx;
  gy = hits[0].gy;
  gz = hits[0].gz;
  p = hits[0].p;
  m = hits[0].m;
  pdg = hits[0].pdg;
  e = 0;
  mintrid = 1000000000;
  maxtrid = -1;
  for (std::vector<MyCSCSimHit>::const_iterator itr = hits.begin(); itr != hits.end(); itr++)
  {
    MyCSCSimHit sh = *itr;
    e += sh.e;
    if (sh.t < mint) mint = sh.t;
    if (sh.t > maxt) maxt = sh.t;
    if (sh.w < minw) minw = sh.w;
    if (sh.w > maxw) maxw = sh.w;
    if (sh.s < mins) mins = sh.s;
    if (sh.s > maxs) maxs = sh.s;
    if (sh.trid < mintrid) mintrid = sh.trid;
    if (sh.trid > maxtrid) maxtrid = sh.trid;
    meant += sh.t;
    sigmat += sh.t*sh.t;
  }
  meant = meant/nh;
  sigmat = sqrt( sigmat/nh - meant*meant);
cout<<" clu: "<<nh<<" "<<mint<<" "<<minw<<" "<<meant<<" "<<r<<" "<<gz<<" "<<m<<" "<<endl;
}


// ================================================================================================
void
MyCSCLayer::init(int l, std::vector<MyCSCCluster> &sclusters)
{
  clusters = sclusters;
  ln = l;
  nclu = clusters.size();
  nh = 0;
  mint = 1000000000.;
  maxt = -1.;
  minw = mins = 1000;
  maxw = maxs = -1;
  mintrid = 1000000000;
  maxtrid = -1;
  if (nclu==0) return;
  for (std::vector<MyCSCCluster>::const_iterator itr = clusters.begin(); itr != clusters.end(); itr++)
  {
    MyCSCCluster cl = *itr;
    nh += cl.nh;
    if (cl.mint < mint) mint = cl.mint;
    if (cl.maxt > maxt) maxt = cl.maxt;
    if (cl.minw < minw) minw = cl.minw;
    if (cl.maxw > maxw) maxw = cl.maxw;
    if (cl.mins < mins) mins = cl.mins;
    if (cl.maxs > maxs) maxs = cl.maxs;
    if (cl.mintrid < mintrid) mintrid = cl.mintrid;
    if (cl.maxtrid > maxtrid) maxtrid = cl.maxtrid;
  }
}


// ================================================================================================
void
MyCSCChamber::init(std::vector<MyCSCLayer> &slayers)
{
  nh = nclu = 0;
  nl = slayers.size();
  mint = 1000000000.;
  maxt = -1.;
  minw = mins = 1000;
  maxw = maxs = -1;
  if (nl==0) return;
  l1 = 7;
  ln = -1;
  for (std::vector<MyCSCLayer>::const_iterator itr = slayers.begin(); itr != slayers.end(); itr++)
  {
    MyCSCLayer la = *itr;
    nh += la.nh;
    nclu += la.nclu;
    if (la.ln < l1) l1 = la.ln;
    if (la.ln > ln) ln = la.ln;
    if (la.mint < mint) mint = la.mint;
    if (la.maxt > maxt) maxt = la.maxt;
    if (la.minw < minw) minw = la.minw;
    if (la.maxw > maxw) maxw = la.maxw;
    if (la.mins < mins) mins = la.mins;
    if (la.maxs > maxs) maxs = la.maxs;

  }
}


// ================================================================================================
void
MyCSCEvent::init(std::vector<MyCSCChamber> &schambers)
{
  nch = schambers.size();
  nh = nclu = 0;
  nch2 = nch3 = nch4 = nch5 = nch6 = 0;
  for (std::vector<MyCSCChamber>::const_iterator itr = schambers.begin(); itr != schambers.end(); itr++)
  {
    MyCSCChamber ch = *itr;
    nh += ch.nh;
    nclu += ch.nclu;
    if (ch.nl>1) nch2++;
    if (ch.nl>2) nch3++;
    if (ch.nl>3) nch4++;
    if (ch.nl>4) nch5++;
    if (ch.nl>5) nch6++;
  }
}



// ================================================================================================
void
MyGEMDetId::init(GEMDetId &id)
{
  reg    = id.region();
  ring   = id.ring();
  st     = id.station();
  layer  = id.layer();
  ch     = id.chamber();
  part   = id.roll();
  t      = type(id);
}


// ================================================================================================
void
MyGEMSimHit::init(PSimHit &sh, const GEMGeometry* gem_g, const ParticleDataTable * pdt)
{
  LocalPoint hitLP = sh.localPosition();
  pdg = sh.particleType();
  m = 0.00051;

  ParticleData const *pdata = 0;
  HepPDT::ParticleID particleType(pdg);
  if (particleType.isValid()) pdata = pdt->particle(particleType);
  if (pdata)  m = pdata->mass();
  //  cout<<"  "<<pdg<<" "<<pdata->name()<<" "<<m;
  else cout<<" pdg not in PDT: "<< pdg<<endl;

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  p = sh.pabs();
  t = sh.tof();
  trid = sh.trackId();

  GlobalPoint hitGP = gem_g->idToDet(DetId(sh.detUnitId()))->surface().toGlobal(hitLP);

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gx = hitGP.x();
  gy = hitGP.y();
  gz = hitGP.z();

  GEMDetId rollId(sh.detUnitId());
  s = gem_g->etaPartition(rollId)->strip(hitLP);
}


bool MyGEMSimHit::operator<(const MyGEMSimHit & rhs) const
{
  // first sort by strip, then by TOF
  if (s==rhs.s) return t<rhs.t;
  else return s<rhs.s;
}


// ================================================================================================
void
MyGEMCluster::init(std::vector<MyGEMSimHit> &shits)
{
  hits = shits;
  nh = hits.size();
  mint = 1000000000.;
  maxt = -1.;
  mins = 1000;
  maxs = -1;
  meant = 0;
  sigmat = 0;
  if (nh==0) return;
  r = hits[0].r;
  eta = hits[0].eta;
  phi = hits[0].phi;
  gx = hits[0].gx;
  gy = hits[0].gy;
  gz = hits[0].gz;
  p = hits[0].p;
  m = hits[0].m;
  pdg = hits[0].pdg;
  e = 0;
  mintrid = 1000000000;
  maxtrid = -1;
  for (std::vector<MyGEMSimHit>::const_iterator itr = hits.begin(); itr != hits.end(); itr++)
  {
    MyGEMSimHit sh = *itr;
    e += sh.e;
    if (sh.t < mint) mint = sh.t;
    if (sh.t > maxt) maxt = sh.t;
    if (sh.s < mins) mins = sh.s;
    if (sh.s > maxs) maxs = sh.s;
    if (sh.trid < mintrid) mintrid = sh.trid;
    if (sh.trid > maxtrid) maxtrid = sh.trid;
    meant += sh.t;
    sigmat += sh.t*sh.t;
  }
  meant = meant/nh;
  sigmat = sqrt( sigmat/nh - meant*meant);
  cout<<" gem clu: "<<nh<<" "<<mint<<" "<<mins<<" "<<meant<<" "<<r<<" "<<gz<<" "<<m<<" "<<endl;
}


// ================================================================================================
void
MyGEMPart::init(int p, int l, std::vector<MyGEMCluster> &sclusters)
{
  clusters = sclusters;
  pn = p;
  ln = l;
  nclu = clusters.size();
  nh = 0;
  mint = 1000000000.;
  maxt = -1.;
  mins = 1000;
  maxs = -1;
  mintrid = 1000000000;
  maxtrid = -1;
  if (nclu==0) return;
  for (std::vector<MyGEMCluster>::const_iterator itr = clusters.begin(); itr != clusters.end(); itr++)
  {
    MyGEMCluster cl = *itr;
    nh += cl.nh;
    if (cl.mint < mint) mint = cl.mint;
    if (cl.maxt > maxt) maxt = cl.maxt;
    if (cl.mins < mins) mins = cl.mins;
    if (cl.maxs > maxs) maxs = cl.maxs;
    if (cl.mintrid < mintrid) mintrid = cl.mintrid;
    if (cl.maxtrid > maxtrid) maxtrid = cl.maxtrid;
  }
}


// ================================================================================================
void
MyGEMChamber::init(std::vector<MyGEMPart> &sparts)
{
  nh = nclu = nl = 0;
  np = sparts.size();
  mint = 1000000000.;
  maxt = -1.;
  if (np==0) return;
  std::set<int> layers;
  for (std::vector<MyGEMPart>::const_iterator itr = sparts.begin(); itr != sparts.end(); itr++)
  {
    MyGEMPart rl = *itr;
    nh += rl.nh;
    nclu += rl.nclu;
    layers.insert(rl.ln);
    if (rl.mint < mint) mint = rl.mint;
    if (rl.maxt > maxt) maxt = rl.maxt;
  }
  nl = layers.size();
}


// ================================================================================================
void
MyGEMEvent::init(std::vector<MyGEMChamber> &schambers)
{
  nch = schambers.size();
  nh = nclu = np = 0;
  for (std::vector<MyGEMChamber>::const_iterator itr = schambers.begin(); itr != schambers.end(); itr++)
  {
    MyGEMChamber ch = *itr;
    nh += ch.nh;
    nclu += ch.nclu;
    np += ch.np;
  }
}


// ================================================================================================
void
MyRPCDetId::init(RPCDetId &id)
{
  reg    = id.region();
  ring   = id.ring();
  st     = id.station();
  sec    = id.sector();
  layer  = id.layer();
  subsec = id.subsector();
  roll   = id.roll();
  t      = type(id);
}


// ================================================================================================
void
MyRPCSimHit::init(PSimHit &sh, const RPCGeometry* rpc_g, const ParticleDataTable * pdt)
{
  LocalPoint hitLP = sh.localPosition();
  pdg = sh.particleType();
  m = 0.00051;

  ParticleData const *pdata = 0;
  HepPDT::ParticleID particleType(pdg);
  if (particleType.isValid()) pdata = pdt->particle(particleType);
  if (pdata)  m = pdata->mass();
  //  cout<<"  "<<pdg<<" "<<pdata->name()<<" "<<m;
  else cout<<" pdg not in PDT: "<< pdg<<endl;

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  p = sh.pabs();
  t = sh.tof();
  trid = sh.trackId();

  GlobalPoint hitGP = rpc_g->idToDet(DetId(sh.detUnitId()))->surface().toGlobal(hitLP);

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gx = hitGP.x();
  gy = hitGP.y();
  gz = hitGP.z();

  RPCDetId rollId(sh.detUnitId());
  s = rpc_g->roll(rollId)->strip(hitLP);
}


bool MyRPCSimHit::operator<(const MyRPCSimHit & rhs) const
{
  // first sort by strip, then by TOF
  if (s==rhs.s) return t<rhs.t;
  else return s<rhs.s;
}


// ================================================================================================
void
MyRPCCluster::init(std::vector<MyRPCSimHit> &shits)
{
  hits = shits;
  nh = hits.size();
  mint = 1000000000.;
  maxt = -1.;
  mins = 1000;
  maxs = -1;
  meant = 0;
  sigmat = 0;
  if (nh==0) return;
  r = hits[0].r;
  eta = hits[0].eta;
  phi = hits[0].phi;
  gx = hits[0].gx;
  gy = hits[0].gy;
  gz = hits[0].gz;
  p = hits[0].p;
  m = hits[0].m;
  pdg = hits[0].pdg;
  e = 0;
  mintrid = 1000000000;
  maxtrid = -1;
  for (std::vector<MyRPCSimHit>::const_iterator itr = hits.begin(); itr != hits.end(); itr++)
  {
    MyRPCSimHit sh = *itr;
    e += sh.e;
    if (sh.t < mint) mint = sh.t;
    if (sh.t > maxt) maxt = sh.t;
    if (sh.s < mins) mins = sh.s;
    if (sh.s > maxs) maxs = sh.s;
    if (sh.trid < mintrid) mintrid = sh.trid;
    if (sh.trid > maxtrid) maxtrid = sh.trid;
    meant += sh.t;
    sigmat += sh.t*sh.t;
  }
  meant = meant/nh;
  sigmat = sqrt( sigmat/nh - meant*meant);
  cout<<" rpc clu: "<<nh<<" "<<mint<<" "<<mins<<" "<<meant<<" "<<r<<" "<<gz<<" "<<m<<" "<<endl;
}


// ================================================================================================
void
MyRPCRoll::init(int r, int l, std::vector<MyRPCCluster> &sclusters)
{
  clusters = sclusters;
  rn = r;
  ln = l;
  nclu = clusters.size();
  nh = 0;
  mint = 1000000000.;
  maxt = -1.;
  mins = 1000;
  maxs = -1;
  mintrid = 1000000000;
  maxtrid = -1;
  if (nclu==0) return;
  for (std::vector<MyRPCCluster>::const_iterator itr = clusters.begin(); itr != clusters.end(); itr++)
  {
    MyRPCCluster cl = *itr;
    nh += cl.nh;
    if (cl.mint < mint) mint = cl.mint;
    if (cl.maxt > maxt) maxt = cl.maxt;
    if (cl.mins < mins) mins = cl.mins;
    if (cl.maxs > maxs) maxs = cl.maxs;
    if (cl.mintrid < mintrid) mintrid = cl.mintrid;
    if (cl.maxtrid > maxtrid) maxtrid = cl.maxtrid;
  }
}


// ================================================================================================
void
MyRPCChamber::init(std::vector<MyRPCRoll> &srolls)
{
  nh = nclu = nl = 0;
  nr = srolls.size();
  mint = 1000000000.;
  maxt = -1.;
  if (nr==0) return;
  std::set<int> layers;
  for (std::vector<MyRPCRoll>::const_iterator itr = srolls.begin(); itr != srolls.end(); itr++)
  {
    MyRPCRoll rl = *itr;
    nh += rl.nh;
    nclu += rl.nclu;
    layers.insert(rl.ln);
    if (rl.mint < mint) mint = rl.mint;
    if (rl.maxt > maxt) maxt = rl.maxt;
  }
  nl = layers.size();
}


// ================================================================================================
void
MyRPCEvent::init(std::vector<MyRPCChamber> &schambers)
{
  nch = schambers.size();
  nh = nclu = nr = 0;
  for (std::vector<MyRPCChamber>::const_iterator itr = schambers.begin(); itr != schambers.end(); itr++)
  {
    MyRPCChamber ch = *itr;
    nh += ch.nh;
    nclu += ch.nclu;
    nr += ch.nr;
  }
}


// ================================================================================================
void
MyDTDetId::init(DTWireId &id)
{
  st     = id.station();
  wh     = id.wheel();
  sec    = id.sector();
  sl     = id.superLayer();
  l      = id.layer();
  wire   = id.wire();
  t      = type(id);
}


// ================================================================================================
void
MyDTSimHit::init(PSimHit &sh, const DTGeometry* dt_g, const ParticleDataTable * pdt)
{
  LocalPoint hitLP = sh.localPosition();
  pdg = sh.particleType();
  m = 0.00051;

  ParticleData const *pdata = 0;
  HepPDT::ParticleID particleType(pdg);
  if (particleType.isValid()) pdata = pdt->particle(particleType);
  if (pdata)  m = pdata->mass();
  //  cout<<"  "<<pdg<<" "<<pdata->name()<<" "<<m;
  else cout<<" pdg not in PDT: "<< pdg<<endl;

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  p = sh.pabs();
  t = sh.tof();
  trid = sh.trackId();

  GlobalPoint hitGP = dt_g->idToDet(DetId(sh.detUnitId()))->surface().toGlobal(hitLP);

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gx = hitGP.x();
  gy = hitGP.y();
  gz = hitGP.z();

  //w = csclayer->geometry()->wireGroup(csclayer->geometry()->nearestWire(hitLP));
  //s = csclayer->geometry()->nearestStrip(hitLP);
}
