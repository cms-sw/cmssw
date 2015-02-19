#include "Validation/MuonRPCDigis/interface/RPCDigiValid.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include <cmath>

using namespace std;
using namespace edm;

RPCDigiValid::RPCDigiValid(const ParameterSet& ps)
{

//  Init the tokens for run data retrieval - stanislav
//  ps.getUntackedParameter<InputTag> retrieves a InputTag from the configuration. The second param is default value
//  module, instance and process labels may be passed in a single string if separated by colon ':'
//  (@see the edm::InputTag constructor documentation)
  simHitToken = consumes<PSimHitContainer>(ps.getUntrackedParameter<edm::InputTag >("simHitTag", edm::InputTag("g4SimHits:MuonRPCHits")));
  rpcDigiToken    = consumes<RPCDigiCollection>(ps.getUntrackedParameter<edm::InputTag>("rpcDigiTag", edm::InputTag("simMuonRPCDigis")));

  outputFile_ = ps.getUntrackedParameter<string> ("outputFile", "rpcDigiValidPlots.root");
}

RPCDigiValid::~RPCDigiValid()
{
}


void RPCDigiValid::analyze(const Event& event, const EventSetup& eventSetup)
{

  // Get the RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord> ().get(rpcGeom);

  edm::Handle<PSimHitContainer> simHit;
  edm::Handle<RPCDigiCollection> rpcDigis;
  event.getByToken(simHitToken, simHit);
  event.getByToken(rpcDigiToken, rpcDigis);

  // Loop on simhits
  PSimHitContainer::const_iterator simIt;

  //loop over Simhit
  std::map<RPCDetId, std::vector<double> > allsims;

  for (simIt = simHit->begin(); simIt != simHit->end(); simIt++)
  {
    RPCDetId Rsid = (RPCDetId)(*simIt).detUnitId();
    const RPCRoll* soll = dynamic_cast<const RPCRoll*> (rpcGeom->roll(Rsid));
    int ptype = simIt->particleType();

    if (ptype == 13 || ptype == -13)
    {

      std::vector<double> buff;
      if (allsims.find(Rsid) != allsims.end())
      {
        buff = allsims[Rsid];
      }

      buff.push_back(simIt->localPosition().x());

      allsims[Rsid] = buff;
    }
    GlobalPoint p = soll->toGlobal(simIt->localPosition());

    double sim_x = p.x();
    double sim_y = p.y();

    xyview->Fill(sim_x, sim_y);

    if (Rsid.region() == (+1))
    {
      if (Rsid.station() == 4)
      {
        xyvDplu4->Fill(sim_x, sim_y);
      }
    }
    else if (Rsid.region() == (-1))
    {
      if (Rsid.station() == 4)
      {
        xyvDmin4->Fill(sim_x, sim_y);
      }
    }
    rzview->Fill(p.z(), p.perp());
  }
  //loop over Digis
  RPCDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = rpcDigis->begin(); detUnitIt != rpcDigis->end(); ++detUnitIt)
  {
    const RPCDetId Rsid = (*detUnitIt).first;
    const RPCRoll* roll = dynamic_cast<const RPCRoll*> (rpcGeom->roll(Rsid));

    const RPCDigiCollection::Range& range = (*detUnitIt).second;
    std::vector<double> sims;
    if (allsims.find(Rsid) != allsims.end())
    {
      sims = allsims[Rsid];
    }

    int ndigi=0;
    for (RPCDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt)
    {
      StripProf->Fill(digiIt->strip());
      BxDist->Fill(digiIt->bx());
      //bx for 4 endcaps
      if (Rsid.region() == (+1))
      {
        if (Rsid.station() == 4)
          BxDisc_4Plus->Fill(digiIt->bx());
      }
      else if (Rsid.region() == (-1))
      {
        if (Rsid.station() == 4)
          BxDisc_4Min->Fill(digiIt->bx());
      }
      ndigi++;
    }

    if (sims.size() == 0)
    {
      noiseCLS->Fill(ndigi);
      if (Rsid.region() == 0)
      {
        noiseCLSBarrel->Fill(ndigi);
      }
      if (Rsid.region() == +1 || Rsid.region() == 1)
      {
        noiseCLSEndcaps->Fill(ndigi);
      }
    }
    //CLS histos
    if (Rsid.region() == 0)
    {
      clsBarrel->Fill(ndigi);
    }
    //endcap 
    if (Rsid.region() != 0)
    {
      if (Rsid.ring() == 2)
      {
        if (abs(Rsid.station()) == 1)
        {
          if (Rsid.roll() == 1)
            CLS_Endcap_1_Ring2_A->Fill(ndigi);
          if (Rsid.roll() == 2)
            CLS_Endcap_1_Ring2_B->Fill(ndigi);
          if (Rsid.roll() == 3)
            CLS_Endcap_1_Ring2_C->Fill(ndigi);
        }
        if (abs(Rsid.station()) == 2 || abs(Rsid.station()) == 3)
        {
          if (Rsid.roll() == 1)
            CLS_Endcap_23_Ring2_A->Fill(ndigi);
          if (Rsid.roll() == 2)
            CLS_Endcap_23_Ring2_B->Fill(ndigi);
          if (Rsid.roll() == 3)
            CLS_Endcap_23_Ring2_C->Fill(ndigi);
        }
      }
      if (Rsid.ring() == 3)
      {
        if (Rsid.roll() == 1)
          CLS_Endcap_123_Ring3_A->Fill(ndigi);
        if (Rsid.roll() == 2)
          CLS_Endcap_123_Ring3_B->Fill(ndigi);
        if (Rsid.roll() == 3)
          CLS_Endcap_123_Ring3_C->Fill(ndigi);
      }
      if (abs(Rsid.station()) == 4)
        CLS_Endcap_4->Fill(ndigi);
    }

    if (sims.size() == 1 && ndigi == 1)
    {
      double dis = roll->centreOfStrip(range.first->strip()).x() - sims[0];
      Res->Fill(dis);

      if (Rsid.region() == 0)
      {
        if (Rsid.ring() == -2)
          ResWmin2->Fill(dis);
        else if (Rsid.ring() == -1)
          ResWmin1->Fill(dis);
        else if (Rsid.ring() == 0)
          ResWzer0->Fill(dis);
        else if (Rsid.ring() == 1)
          ResWplu1->Fill(dis);
        else if (Rsid.ring() == 2)
          ResWplu2->Fill(dis);
        //barrel layers
        if (Rsid.station() == 1 && Rsid.layer() == 1)
          ResLayer1_barrel->Fill(dis);
        if (Rsid.station() == 1 && Rsid.layer() == 2)
          ResLayer2_barrel->Fill(dis);
        if (Rsid.station() == 2 && Rsid.layer() == 1)
          ResLayer3_barrel->Fill(dis);
        if (Rsid.station() == 2 && Rsid.layer() == 2)
          ResLayer4_barrel->Fill(dis);
        if (Rsid.station() == 3)
          ResLayer5_barrel->Fill(dis);
        if (Rsid.station() == 4)
          ResLayer6_barrel->Fill(dis);
      }
      //endcap layers residuals
      if (Rsid.region() != 0)
      {
        if (Rsid.ring() == 2)
        {
          if (abs(Rsid.station()) == 1)
          {
            if (Rsid.roll() == 1)
              Res_Endcap1_Ring2_A->Fill(dis);
            if (Rsid.roll() == 2)
              Res_Endcap1_Ring2_B->Fill(dis);
            if (Rsid.roll() == 3)
              Res_Endcap1_Ring2_C->Fill(dis);
          }
          if (abs(Rsid.station()) == 2 || abs(Rsid.station()) == 3)
          {
            if (Rsid.roll() == 1)
              Res_Endcap23_Ring2_A->Fill(dis);
            if (Rsid.roll() == 2)
              Res_Endcap23_Ring2_B->Fill(dis);
            if (Rsid.roll() == 3)
              Res_Endcap23_Ring2_C->Fill(dis);
          }
        }
        if (Rsid.ring() == 3)
        {
          if (Rsid.roll() == 1)
            Res_Endcap123_Ring3_A->Fill(dis);
          if (Rsid.roll() == 2)
            Res_Endcap123_Ring3_B->Fill(dis);
          if (Rsid.roll() == 3)
            Res_Endcap123_Ring3_C->Fill(dis);
        }
      }

      if (Rsid.region() == (+1))
      {

        if (Rsid.station() == 1)
          ResDplu1->Fill(dis);
        else if (Rsid.station() == 2)
          ResDplu2->Fill(dis);
        else if (Rsid.station() == 3)
          ResDplu3->Fill(dis);
        else if (Rsid.station() == 4)
          ResDplu4->Fill(dis);
      }
      if (Rsid.region() == (-1))
      {

        if (Rsid.station() == 1)
          ResDmin1->Fill(dis);
        else if (Rsid.station() == 2)
          ResDmin2->Fill(dis);
        else if (Rsid.station() == 3)
          ResDmin3->Fill(dis);
        else if (Rsid.station() == 4)
          ResDmin4->Fill(dis);
      }
    }
  }
}

void RPCDigiValid::bookHistograms(DQMStore::IBooker& booker, edm::Run const& run, edm::EventSetup const& eSetup)
{
  booker.setCurrentFolder("RPCDigisV/RPCDigis");

  xyview = booker.book2D("X_Vs_Y_View", "X_Vs_Y_View", 155, -775., 775., 155, -775., 775.);

  xyvDplu4 = booker.book2D("Dplu4_XvsY", "Dplu4_XvsY", 155, -775., 775., 155, -775., 775.);
  xyvDmin4 = booker.book2D("Dmin4_XvsY", "Dmin4_XvsY", 155, -775., 775., 155, -775., 775.);

  rzview = booker.book2D("R_Vs_Z_View", "R_Vs_Z_View", 216, -1080., 1080., 52, 260., 780.);
  Res = booker.book1D("Digi_SimHit_difference", "Digi_SimHit_difference", 300, -8, 8);
  ResWmin2 = booker.book1D("W_Min2_Residuals", "W_Min2_Residuals", 400, -8, 8);
  ResWmin1 = booker.book1D("W_Min1_Residuals", "W_Min1_Residuals", 400, -8, 8);
  ResWzer0 = booker.book1D("W_Zer0_Residuals", "W_Zer0_Residuals", 400, -8, 8);
  ResWplu1 = booker.book1D("W_Plu1_Residuals", "W_Plu1_Residuals", 400, -8, 8);
  ResWplu2 = booker.book1D("W_Plu2_Residuals", "W_Plu2_Residuals", 400, -8, 8);

  ResLayer1_barrel = booker.book1D("ResLayer1_barrel", "ResLayer1_barrel", 400, -8, 8);
  ResLayer2_barrel = booker.book1D("ResLayer2_barrel", "ResLayer2_barrel", 400, -8, 8);
  ResLayer3_barrel = booker.book1D("ResLayer3_barrel", "ResLayer3_barrel", 400, -8, 8);
  ResLayer4_barrel = booker.book1D("ResLayer4_barrel", "ResLayer4_barrel", 400, -8, 8);
  ResLayer5_barrel = booker.book1D("ResLayer5_barrel", "ResLayer5_barrel", 400, -8, 8);
  ResLayer6_barrel = booker.book1D("ResLayer6_barrel", "ResLayer6_barrel", 400, -8, 8);

  BxDist = booker.book1D("Bunch_Crossing", "Bunch_Crossing", 20, -10., 10.);
  StripProf = booker.book1D("Strip_Profile", "Strip_Profile", 100, 0, 100);

  BxDisc_4Plus = booker.book1D("BxDisc_4Plus", "BxDisc_4Plus", 20, -10., 10.);
  BxDisc_4Min = booker.book1D("BxDisc_4Min", "BxDisc_4Min", 20, -10., 10.);

  //cls histos
  noiseCLS = booker.book1D("noiseCLS", "noiseCLS", 10, 0.5, 10.5);
  noiseCLSBarrel = booker.book1D("noiseCLSBarrel", "noiseCLSBarrel", 10, 0.5, 10.5);
  noiseCLSEndcaps = booker.book1D("noiseCLSEndcaps", "noiseCLSEndcaps", 10, 0.5, 10.5);

  clsBarrel = booker.book1D("clsBarrel", "clsBarrel", 10, 0.5, 10.5);

  //endcap CLS
  CLS_Endcap_1_Ring2_A = booker.book1D("CLS_Endcap_1_1Ring2_A", "CLS_Endcap_1_Ring2_A", 10, 0.5, 10.5);
  CLS_Endcap_1_Ring2_B = booker.book1D("CLS_Endcap_1_1Ring2_B", "CLS_Endcap_1_Ring2_B", 10, 0.5, 10.5);
  CLS_Endcap_1_Ring2_C = booker.book1D("CLS_Endcap_1_1Ring2_C", "CLS_Endcap_1_Ring2_C", 10, 0.5, 10.5);

  CLS_Endcap_23_Ring2_A = booker.book1D("CLS_Endcap_23_Ring2_A", "CLS_Endcap_23_Ring2_A", 10, 0.5, 10.5);
  CLS_Endcap_23_Ring2_B = booker.book1D("CLS_Endcap_23_Ring2_B", "CLS_Endcap_23_Ring2_B", 10, 0.5, 10.5);
  CLS_Endcap_23_Ring2_C = booker.book1D("CLS_Endcap_23_Ring2_C", "CLS_Endcap_23_Ring2_C", 10, 0.5, 10.5);

  CLS_Endcap_123_Ring3_A = booker.book1D("CLS_Endcap_123_Ring3_A", "CLS_Endcap_123_Ring3_A", 10, 0.5, 10.5);
  CLS_Endcap_123_Ring3_B = booker.book1D("CLS_Endcap_123_Ring3_B", "CLS_Endcap_123_Ring3_B", 10, 0.5, 10.5);
  CLS_Endcap_123_Ring3_C = booker.book1D("CLS_Endcap_123_Ring3_C", "CLS_Endcap_123_Ring3_C", 10, 0.5, 10.5);

  CLS_Endcap_4 = booker.book1D("CLS_Endcap_4", "CLS_Endcap_4", 10, 0.5, 10.5);

  //endcap residuals
  ResDmin1 = booker.book1D("Disk_Min1_Residuals", "Disk_Min1_Residuals", 400, -8, 8);
  ResDmin2 = booker.book1D("Disk_Min2_Residuals", "Disk_Min2_Residuals", 400, -8, 8);
  ResDmin3 = booker.book1D("Disk_Min3_Residuals", "Disk_Min3_Residuals", 400, -8, 8);
  ResDplu1 = booker.book1D("Disk_Plu1_Residuals", "Disk_Plu1_Residuals", 400, -8, 8);
  ResDplu2 = booker.book1D("Disk_Plu2_Residuals", "Disk_Plu2_Residuals", 400, -8, 8);
  ResDplu3 = booker.book1D("Disk_Plu3_Residuals", "Disk_Plu3_Residuals", 400, -8, 8);

  ResDmin4 = booker.book1D("Disk_Min4_Residuals", "Disk_Min4_Residuals", 400, -8, 8);
  ResDplu4 = booker.book1D("Disk_Plu4_Residuals", "Disk_Plu4_Residuals", 400, -8, 8);

  Res_Endcap1_Ring2_A = booker.book1D("Res_Endcap1_Ring2_A", "Res_Endcap1_Ring2_A", 400, -8, 8);
  Res_Endcap1_Ring2_B = booker.book1D("Res_Endcap1_Ring2_B", "Res_Endcap1_Ring2_B", 400, -8, 8);
  Res_Endcap1_Ring2_C = booker.book1D("Res_Endcap1_Ring2_C", "Res_Endcap1_Ring2_C", 400, -8, 8);

  Res_Endcap23_Ring2_A = booker.book1D("Res_Endcap23_Ring2_A", "Res_Endcap23_Ring2_A", 400, -8, 8);
  Res_Endcap23_Ring2_B = booker.book1D("Res_Endcap23_Ring2_B", "Res_Endcap23_Ring2_B", 400, -8, 8);
  Res_Endcap23_Ring2_C = booker.book1D("Res_Endcap23_Ring2_C", "Res_Endcap23_Ring2_C", 400, -8, 8);

  Res_Endcap123_Ring3_A = booker.book1D("Res_Endcap123_Ring3_A", "Res_Endcap123_Ring3_A", 400, -8, 8);
  Res_Endcap123_Ring3_B = booker.book1D("Res_Endcap123_Ring3_B", "Res_Endcap123_Ring3_B", 400, -8, 8);
  Res_Endcap123_Ring3_C = booker.book1D("Res_Endcap123_Ring3_C", "Res_Endcap123_Ring3_C", 400, -8, 8);
}

