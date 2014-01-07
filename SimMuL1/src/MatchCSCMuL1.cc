// -*- C++ -*-
//
// Package:    SimMuL1
// Class:      MatchCSCMuL1
// 
/**\class MatchCSCMuL1 MatchCSCMuL1.cc MyCode/SimMuL1/src/MatchCSCMuL1.cc

Description: Trigger Matching info for SimTrack in CSC

Implementation:
<Notes on implementation>
 */
//
// Original Author:  "Vadim Khotilovich"
//         Created:  Mon May  5 20:50:43 CDT 2008
// $Id$
//
//


#include "GEMCode/SimMuL1/interface/MatchCSCMuL1.h"

// system include files
#include <memory>
#include <cmath>
#include <set>

#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>


namespace 
{
    const double MUON_MASS = 0.105658369 ; // PDG06

    double ptscale[33] = { 
        -1.,   0.0,   1.5,   2.0,   2.5,   3.0,   3.5,   4.0,
        4.5,   5.0,   6.0,   7.0,   8.0,  10.0,  12.0,  14.0,  
        16.0,  18.0,  20.0,  25.0,  30.0,  35.0,  40.0,  45.0, 
        50.0,  60.0,  70.0,  80.0,  90.0, 100.0, 120.0, 140.0, 1.E6
    };

}


//_____________________________________________________________________________
// Constructor
MatchCSCMuL1::MatchCSCMuL1(const SimTrack  *s, const SimVertex *v, const CSCGeometry* g):
    strk(s), svtx(v), cscGeometry(g)
{
    double endcap = (strk->momentum().eta() >= 0) ? 1. : -1.;
    math::XYZVectorD v0(0.000001,0.,endcap);
    pME11 = pME1 = pME2 = pME3 = v0;
}


//_____________________________________________________________________________
/*
 * Add a simhit to the vector of simhits, hitsMapLayer and hitsMapChamber
 *
 * @param h   The simhit
 */
    void 
MatchCSCMuL1::addSimHit(PSimHit & h)
{
    simHits.push_back(h);
    hitsMapLayer[h.detUnitId()].push_back(h);
    CSCDetId layerId( h.detUnitId() );
    hitsMapChamber[layerId.chamberId().rawId()].push_back(h);
}


//_____________________________________________________________________________
    int 
MatchCSCMuL1::keyStation()
{
    // first try ME2
    if (pME2.Rho()>0.01) return 2;
    // next, try ME3
    if (pME3.Rho()>0.01) return 3;
    // last, try ME1
    if (pME1.Rho()>0.01) return 1;
    return 99;
}


//_____________________________________________________________________________
/*
 * Return the position vector of the propagated simtrack in ME11, ME1, ME2 or ME3
 *
 * @param st   The station number
 * @return     The position vector of the propagated simtrack 
 */
    math::XYZVectorD
MatchCSCMuL1::vAtStation(int st)
{
    switch (st){
        case 0: return pME11; break;
        case 1: return pME1; break;
        case 2: return pME2; break;
        case 3: return pME3; break;
    }
    double endcap = (strk->momentum().eta() >= 0) ? 1. : -1.;
    math::XYZVectorD v0(0.000001,0.,endcap);
    return v0;
}


//_____________________________________________________________________________
    math::XYZVectorD
MatchCSCMuL1::vSmart()
{
    return vAtStation(keyStation());
}


//_____________________________________________________________________________
    double
MatchCSCMuL1::deltaRAtStation(int station, double to_eta, double to_phi)
{
    math::XYZVectorD v = vAtStation(station);
    if (v.Rho()<0.01) return 999.;
    return deltaR(v.eta(),normalizedPhi(v.phi()), to_eta,to_phi);
}


//_____________________________________________________________________________
    double
MatchCSCMuL1::deltaRSmart(double to_eta, double to_phi)
{
    return deltaRAtStation(keyStation(), to_eta,to_phi);
}


//_____________________________________________________________________________
/*
 * Return the number of simhits in the event
 * Option to count only muon simhits 
 */
    int 
MatchCSCMuL1::nSimHits()
{
    if (!muOnly) return simHits.size();
    int n=0;
    for (unsigned j=0; j<simHits.size(); j++) if (abs(simHits[j].particleType())==13 ) n++;
    return n;
}


//_____________________________________________________________________________
/*
 * Return the detIds with simhits
 * Option to count only detIds with muon simhits
 */
    std::vector<int> 
MatchCSCMuL1::detsWithHits()
{
    std::set<int> dets;
    std::map<int, std::vector<PSimHit> >::const_iterator mapItr = hitsMapLayer.begin();
    for( ; mapItr != hitsMapLayer.end(); ++mapItr) 
        if ( !muOnly || abs((mapItr->second)[0].particleType())==13 ) 
            dets.insert(mapItr->first);
    return std::vector<int>(dets.begin(), dets.end()); 
}


//_____________________________________________________________________________
/*
 * Return the chambers with hits in a particular station, ring and with a 
 * minimum number of simhits
 *
 * @param station    The station number
 * @param ring       The ring number - if 0, it will look in all rings. 
 * @param minNHits   Minimum number of simhits
 * @return           The chamber numbers
 */
    std::vector<int> 
MatchCSCMuL1::chambersWithHits(int station, int ring, unsigned minNHits)
{
    std::set<int> chambers;
    std::set<int> layersWithHits;

    std::map<int, std::vector<PSimHit> >::const_iterator mapItr = hitsMapChamber.begin();
    for( ; mapItr != hitsMapChamber.end(); ++mapItr){
        CSCDetId cid(mapItr->first);
        if (station && cid.station() != station) continue;
        if (ring && cid.ring() != ring) continue;
        layersWithHits.clear();
        for (unsigned i=0; i<(mapItr->second).size(); i++)
            if ( !muOnly || abs((mapItr->second)[i].particleType())==13 ) {
                CSCDetId hid((mapItr->second)[i].detUnitId());
                layersWithHits.insert(hid.layer());
            }
        if (layersWithHits.size()>=minNHits) chambers.insert( mapItr->first );
        // if chamber is ME1b then add layers with hits in corresponding ME1a
        if (cid.station() == 1 && cid.ring() == 1) {
            CSCDetId ME1aId(cid.endcap(),cid.station(),4,cid.chamber(),0);
            if (layersWithHits.size()+
                    numberOfLayersWithHitsInChamber(ME1aId.rawId())
                    >= minNHits) chambers.insert( mapItr->first );
        }
    }
    std::vector<int> result( chambers.begin() , chambers.end() );
    return result;
}


//_____________________________________________________________________________
/*
 * Return the simhits for a particular detId
 * INFO: this function is in use
 */
    std::vector<PSimHit>
MatchCSCMuL1::layerHits(int detId)
{
    std::vector<PSimHit> result;
    std::map<int, std::vector<PSimHit> >::const_iterator mapItr = hitsMapLayer.find(detId);
    if (mapItr == hitsMapLayer.end()) return result;
    for (unsigned i=0; i<(mapItr->second).size(); i++)
        if ( !muOnly || abs((mapItr->second)[i].particleType())==13 ) 
            result.push_back((mapItr->second)[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the simhits for a particular detId
 */
    std::vector<PSimHit>
MatchCSCMuL1::chamberHits(int detId)
{
    // foolproof chamber id
    CSCDetId dId(detId);
    CSCDetId chamberId = dId.chamberId();

    std::vector<PSimHit> result;
    std::map<int, std::vector<PSimHit> >::const_iterator mapItr = hitsMapChamber.find(chamberId);
    if (mapItr == hitsMapChamber.end()) return result;
    for (unsigned i=0; i<(mapItr->second).size(); i++)
        if ( !muOnly || abs((mapItr->second)[i].particleType())==13 ) 
            result.push_back((mapItr->second)[i]);

    return result;
}


//_____________________________________________________________________________
/*
 * Return all simhits in the event
 * Option to return only muon simhits
 */
    std::vector<PSimHit> 
MatchCSCMuL1::allSimHits()
{
    if (!muOnly) return simHits;
    std::vector<PSimHit> result;
    for (unsigned j=0; j<simHits.size(); j++) 
        if (abs(simHits[j].particleType())==13 ) 
            result.push_back(simHits[j]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the number of layers with a minimum number of hits in the chamber
 * The minimum number of layers is specified elsewhere
 * There is an option to only consider muon simhits in the counting of the layers
 * 
 * @param detId   The detId
 * @return        The number of layers with hits
 */
    int
MatchCSCMuL1::numberOfLayersWithHitsInChamber(int detId)
{
    std::set<int> layersWithHits;

    std::vector<PSimHit> chHits = chamberHits(detId);
    for (unsigned i = 0; i < chHits.size(); i++) 
    {
        CSCDetId hid(chHits[i].detUnitId());
        if ( !muOnly || abs(chHits[i].particleType())==13 ) 
            layersWithHits.insert(hid.layer());
    }
    return layersWithHits.size();
}


//_____________________________________________________________________________
    std::pair<int,int>
MatchCSCMuL1::wireGroupAndStripInChamber( int detId )
{
    std::pair<int,int> err_pair(-1,-1);

    std::vector<PSimHit> hits = chamberHits( detId );
    unsigned n = hits.size();
    if ( n == 0 ) return err_pair;

    if (CSCConstants::KEY_CLCT_LAYER != CSCConstants::KEY_ALCT_LAYER)  std::cout<<"ALARM: KEY_CLCT_LAYER != KEY_ALCT_LAYER"<<std::endl;

    // find LocalPoint of the highest energy muon simhit in key layer
    // if no hit in key layer, take the highest energy muon simhit local position
    LocalPoint lpkey(0.,0.,0.), lphe(0.,0.,0.);
    double elosskey=-1., eloss=-1.;
    for (unsigned i = 0; i < n; i++)
    {
        CSCDetId lid(hits[i].detUnitId());
        double el = hits[i].energyLoss();
        if ( el > eloss ) {
            lphe = hits[i].localPosition();
            eloss = el;
        }
        if (lid.layer() != CSCConstants::KEY_ALCT_LAYER) continue;
        if ( el > elosskey ) {
            lpkey = hits[i].localPosition();
            elosskey = el;
        }
    }
    LocalPoint theLP = lpkey;
    if (elosskey<0.) theLP = lphe;
    const LocalPoint cLP = theLP;

    CSCDetId keyID(detId + CSCConstants::KEY_ALCT_LAYER);
    const CSCLayer* csclayer = cscGeometry->layer(keyID);
    int hitWireG = csclayer->geometry()->wireGroup(csclayer->geometry()->nearestWire(cLP));
    int hitStrip = csclayer->geometry()->nearestStrip(cLP);

    std::pair<int,int> ws(hitWireG,hitStrip);
    return ws;
}


//_____________________________________________________________________________
/*
 * Has the simtrack at least one chamber with the minimum number of hits 
 * 
 * @param st        The station number
 * @param minNHits  The minimum number of hits
 * @return          True if it has a least one such chamber
 */
    bool
MatchCSCMuL1::hasHitsInStation(int st, int ri, unsigned minNHits) // st=0 - any,  st=1,2,3,4 - ME1-4
{
    std::vector<int> chIds = chambersWithHits(st,ri,minNHits);
    return chIds.size()!=0;
}


//_____________________________________________________________________________
/*
 * Get the number of stations with the minimum number of hits
 * @param me1       Station 1 flag: default is true        
 * @param me2       Station 2 flag: default is true     
 * @param me3       Station 3 flag: default is true     
 * @param me4       Station 4 flag: default is true     
 * @param minNHits  The minimum number of hits
 * @return          The number of stations
 */
    unsigned
MatchCSCMuL1::nStationsWithHits(bool me1, bool me2, bool me3, bool me4, unsigned minNHits)
{
    return ( (me1 & hasHitsInStation(1,0,minNHits))
            + (me2 & hasHitsInStation(2,0,minNHits))
            + (me3 & hasHitsInStation(3,0,minNHits))
            + (me4 & hasHitsInStation(4,0,minNHits)) );
}


//_____________________________________________________________________________
/*
 * Return the ALCTs in the readout 
 */
    std::vector< MatchCSCMuL1::ALCT > 
MatchCSCMuL1::ALCTsInReadOut()
{
    std::vector<ALCT> result;
    for (unsigned i=0; i<ALCTs.size();i++) 
        if ( ALCTs[i].inReadOut() ) result.push_back(ALCTs[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the ALCTs in the readout 
 */
    std::vector< MatchCSCMuL1::ALCT >
MatchCSCMuL1::vALCTs(bool readout)
{
    if (readout) return ALCTsInReadOut();
    return ALCTs;
}


//_____________________________________________________________________________
/*
 * Return the chamber Ids with ALCTs 
 * @param readout   Do ALCTs have to be in the readout BX window?
 */
    std::vector<int>
MatchCSCMuL1::chambersWithALCTs(bool readout)
{
    std::vector<ALCT> tALCTs = vALCTs(readout);
    std::set<int> chambers;
    for (unsigned i=0; i<tALCTs.size();i++) 
        chambers.insert( tALCTs[i].id.rawId() );
    return std::vector<int>(chambers.begin(), chambers.end());
}


//_____________________________________________________________________________
/*
 * Return the ALCTs for a particular detId
 * @param detId     The detId
 * @param readout   Do ALCTs have to be in the readout BX window?
 */
    std::vector<MatchCSCMuL1::ALCT>
MatchCSCMuL1::chamberALCTs( int detId, bool readout )
{
    std::vector<ALCT> tALCTs = vALCTs(readout);
    // foolproof chamber id
    CSCDetId dId(detId);
    CSCDetId chamberId = dId.chamberId();
    std::vector<ALCT> result;
    for (unsigned i=0; i<tALCTs.size();i++) 
        if ( tALCTs[i].id == chamberId ) result.push_back(tALCTs[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the BXs with ALCTs for a particular detId
 * @param detId     The detId
 * @param readout   Do ALCTs have to be in the readout BX window?
 */
    std::vector<int> 
MatchCSCMuL1::bxsWithALCTs( int detId, bool readout )
{
    std::vector<MatchCSCMuL1::ALCT> v = chamberALCTs( detId, readout );
    std::set<int> bxs;
    for (unsigned i=0; i<v.size();i++) 
        bxs.insert( v[i].getBX() );
    return std::vector<int>(bxs.begin(), bxs.end());
}


//_____________________________________________________________________________
/*
 * Return the ALCTs in a particular chamber for a particular BX
 * Specify whether the ALCTs have to be in the readout
 * @param detId     The detId
 * @param bx        The BX
 * @param readout   Do ALCTs have to be in the readout BX window?
 * 
 */
    std::vector<MatchCSCMuL1::ALCT> 
MatchCSCMuL1::chamberALCTsInBx( int detId, int bx, bool readout )
{
    std::vector<MatchCSCMuL1::ALCT> v = chamberALCTs( detId, readout );
    std::vector<MatchCSCMuL1::ALCT> result;
    for (unsigned i=0; i<v.size();i++)
        if ( v[i].getBX() == bx ) result.push_back(v[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the CLCTs in the readout 
 */
    std::vector< MatchCSCMuL1::CLCT >
MatchCSCMuL1::CLCTsInReadOut()
{
    std::vector<CLCT> result;
    for (unsigned i=0; i<CLCTs.size();i++)
        if ( CLCTs[i].inReadOut() ) result.push_back(CLCTs[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the CLCTs in the readout 
 */
    std::vector< MatchCSCMuL1::CLCT >
MatchCSCMuL1::vCLCTs(bool readout)
{
    if (readout) return CLCTsInReadOut();
    return CLCTs;
}


//_____________________________________________________________________________
/*
 * Return the chamber Ids with CLCTs 
 * @param readout   Do CLCTs have to be in the readout BX window?
 */
    std::vector<int>
MatchCSCMuL1::chambersWithCLCTs( bool readout)
{
    std::vector<CLCT> tCLCTs = vCLCTs(readout);
    std::set<int> chambers;
    for (unsigned i=0; i<tCLCTs.size();i++) chambers.insert( tCLCTs[i].id.rawId() );
    return std::vector<int>(chambers.begin(), chambers.end());
}


//_____________________________________________________________________________
/*
 * Return the CLCTs for a particular detId
 * @param detId     The detId
 * @param readout   Do CLCTs have to be in the readout BX window?
 */
    std::vector<MatchCSCMuL1::CLCT>
MatchCSCMuL1::chamberCLCTs( int detId, bool readout )
{
    std::vector<CLCT> tCLCTs = vCLCTs(readout);
    // foolproof chamber id
    CSCDetId dId(detId);
    CSCDetId chamberId = dId.chamberId();
    std::vector<CLCT> result;
    for (unsigned i=0; i<tCLCTs.size();i++) 
        if ( tCLCTs[i].id == chamberId ) result.push_back(tCLCTs[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the BXs with CLCTs for a particular detId
 * @param detId     The detId
 * @param readout   Do CLCTs have to be in the readout BX window?
 */
    std::vector<int>
MatchCSCMuL1::bxsWithCLCTs( int detId, bool readout )
{
    std::vector<MatchCSCMuL1::CLCT> v = chamberCLCTs( detId, readout );
    std::set<int> bxs;
    for (unsigned i=0; i<v.size();i++) bxs.insert( v[i].getBX() );
    return std::vector<int>(bxs.begin(), bxs.end());
}


//_____________________________________________________________________________
/*
 * Return the BXs with CLCTs for a particular detId
 * @param detId     The detId
 * @param readout   Do CLCTs have to be in the readout BX window?
 */
    std::vector<MatchCSCMuL1::CLCT>
MatchCSCMuL1::chamberCLCTsInBx( int detId, int bx, bool readout )
{
    std::vector<MatchCSCMuL1::CLCT> v = chamberCLCTs( detId, readout );
    std::vector<MatchCSCMuL1::CLCT> result;
    for (unsigned i=0; i<v.size();i++)
        if ( v[i].getBX() == bx ) result.push_back(v[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the LCTs in the readout 
 */
    std::vector< MatchCSCMuL1::LCT >
MatchCSCMuL1::LCTsInReadOut()
{
    std::vector<LCT> result;
    for (unsigned i=0; i<LCTs.size();i++)
        if ( LCTs[i].inReadOut() ) result.push_back(LCTs[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the LCTs in the readout 
 */
    std::vector< MatchCSCMuL1::LCT >
MatchCSCMuL1::vLCTs(bool readout)
{
    if (readout) return LCTsInReadOut();
    return LCTs;
}


//_____________________________________________________________________________
/*
 * Return the chamber Ids with LCTs 
 * @param readout   Do LCTs have to be in the readout BX window?
 */
    std::vector<int>
MatchCSCMuL1::chambersWithLCTs( bool readout )
{
    std::vector<LCT> tLCTs = vLCTs(readout);
    std::set<int> chambers;
    for (unsigned i=0; i<tLCTs.size();i++) chambers.insert( tLCTs[i].id.rawId() );
    return std::vector<int>(chambers.begin(), chambers.end());
}


//_____________________________________________________________________________
/*
 * Return the LCTs for a particular detId
 * @param detId     The detId
 * @param readout   Do LCTs have to be in the readout BX window?
 */
    std::vector<MatchCSCMuL1::LCT>
MatchCSCMuL1::chamberLCTs( int detId, bool readout )
{
    std::vector<LCT> tLCTs = vLCTs(readout);
    // foolproof chamber id
    CSCDetId dId(detId);
    CSCDetId chamberId = dId.chamberId();
    std::vector<LCT> result;
    for (unsigned i=0; i<tLCTs.size();i++) 
        if ( tLCTs[i].id == chamberId ) result.push_back(tLCTs[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the LCTs for a particular detId
 * @param detId     The detId
 * @param readout   Do LCTs have to be in the readout BX window?
 */
    std::vector<MatchCSCMuL1::LCT*>
MatchCSCMuL1::chamberLCTsp( int detId, bool readout )
{
    // foolproof chamber id
    CSCDetId dId(detId);
    CSCDetId chamberId = dId.chamberId();
    std::vector<LCT*> result;
    for (unsigned i=0; i<LCTs.size();i++) {
        if ( readout && !(LCTs[i].inReadOut()) ) continue;
        if ( LCTs[i].id == chamberId ) result.push_back( &(LCTs[i]) );
    }
    return result;
}


//_____________________________________________________________________________
/*
 * Return the BXs with LCTs for a particular detId
 * @param detId     The detId
 * @param readout   Do LCTs have to be in the readout BX window?
 */
    std::vector<int>
MatchCSCMuL1::bxsWithLCTs( int detId, bool readout )
{
    std::vector<MatchCSCMuL1::LCT> v = chamberLCTs( detId, readout );
    std::set<int> bxs;
    for (unsigned i=0; i<v.size();i++) 
        bxs.insert( v[i].getBX() );
    return std::vector<int>(bxs.begin(), bxs.end());
}


//_____________________________________________________________________________
/*
 * Return the LCTs in a particular chamber for a particular BX
 * Specify whether the LCTs have to be in the readout
 * @param detId     The detId
 * @param bx        The BX
 * @param readout   Do LCTs have to be in the readout BX window?
 * 
 */
    std::vector<MatchCSCMuL1::LCT>
MatchCSCMuL1::chamberLCTsInBx( int detId, int bx, bool readout )
{
    std::vector<MatchCSCMuL1::LCT> v = chamberLCTs( detId, readout );
    std::vector<MatchCSCMuL1::LCT> result;
    for (unsigned i=0; i<v.size();i++)
        if ( v[i].getBX() == bx ) result.push_back(v[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the MPC LCTs in the readout 
 */
    std::vector< MatchCSCMuL1::MPLCT >
MatchCSCMuL1::MPLCTsInReadOut()
{
    std::vector<MPLCT> result;
    for (unsigned i=0; i<MPLCTs.size();i++)
        if ( MPLCTs[i].inReadOut() ) result.push_back(MPLCTs[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the MPC LCTs in the readout 
 */
    std::vector< MatchCSCMuL1::MPLCT >
MatchCSCMuL1::vMPLCTs(bool readout)
{
    if (readout) return MPLCTsInReadOut();
    return MPLCTs;
}


//_____________________________________________________________________________
/*
 * Return the chamber Ids with MPC LCTs 
 * @param readout   Do MPC LCTs have to be in the readout BX window?
 */
    std::vector<int>
MatchCSCMuL1::chambersWithMPLCTs(bool readout)
{
    std::vector<MPLCT> tMPLCTs = vMPLCTs(readout);
    std::set<int> chambers;
    for (unsigned i=0; i<tMPLCTs.size();i++) chambers.insert( tMPLCTs[i].id.rawId() );
    return std::vector<int>(chambers.begin(), chambers.end());
}


//_____________________________________________________________________________
/*
 * Return the MPC LCTs for a particular detId
 * @param detId     The detId
 * @param readout   Do MPC LCTs have to be in the readout BX window?
 */
    std::vector<MatchCSCMuL1::MPLCT>
MatchCSCMuL1::chamberMPLCTs( int detId, bool readout )
{
    std::vector<MPLCT> tMPLCTs = vMPLCTs(readout);
    // foolproof chamber id
    CSCDetId dId(detId);
    CSCDetId chamberId = dId.chamberId();
    std::vector<MPLCT> result;
    for (unsigned i=0; i<tMPLCTs.size();i++) 
        if ( tMPLCTs[i].id == chamberId ) result.push_back(tMPLCTs[i]);
    return result;
}


//_____________________________________________________________________________
/*
 * Return the BXs with MPC LCTs for a particular detId
 * @param detId     The detId
 * @param readout   Do MPC LCTs have to be in the readout BX window?
 */
    std::vector<int>
MatchCSCMuL1::bxsWithMPLCTs( int detId, bool readout )
{
    std::vector<MatchCSCMuL1::MPLCT> v = chamberMPLCTs( detId, readout );
    std::set<int> bxs;
    for (unsigned i=0; i<v.size();i++) bxs.insert( v[i].trgdigi->getBX() );
    std::vector<int> result( bxs.begin() , bxs.end() );
    return result;
}


//_____________________________________________________________________________
/*
 * Return the MPC LCTs in a particular chamber for a particular BX
 * Specify whether the MPC LCTs have to be in the readout
 * @param detId     The detId
 * @param bx        The BX
 * @param readout   Do MPC LCTs have to be in the readout BX window?
 * 
 */
    std::vector<MatchCSCMuL1::MPLCT>
MatchCSCMuL1::chamberMPLCTsInBx( int detId, int bx, bool readout )
{
    std::vector<MatchCSCMuL1::MPLCT> v = chamberMPLCTs( detId, readout );
    std::vector<MatchCSCMuL1::MPLCT> result;
    for (unsigned i=0; i<v.size();i++)
        if ( v[i].trgdigi->getBX() == bx ) result.push_back(v[i]);
    return result;
}


//_____________________________________________________________________________
    void
MatchCSCMuL1::print (const char msg[300], bool psimtr, bool psimh,
        bool palct, bool pclct, bool plct, bool pmplct,
        bool ptftrack, bool ptfcand)
{
    std::cout<<"####### MATCH PRINT: "<<msg<<" #######"<<std::endl;

    bool DETAILED_HIT_LAYERS = 0;

    if (psimtr) 
    {
        std::cout<<"****** SimTrack: id="<<strk->trackId()<<"  pt="<<sqrt(strk->momentum().perp2())
            <<"  eta="<<strk->momentum().eta()<<"  phi="<<normalizedPhi( strk->momentum().phi()) 
            <<"   nSimHits="<<simHits.size()<<std::endl;
        std::cout<<"                 nALCT="<<ALCTs.size()<<"    nCLCT="<<CLCTs.size()<<"    nLCT="<<LCTs.size()<<"    nMPLCT="<<MPLCTs.size()
            <<"    nTFTRACK="<<TFTRACKs.size()<<"    nTFCAND="<<TFCANDs.size()<<std::endl;

        int nALCTok=0,nCLCTok=0,nLCTok=0,nMPLCTok=0,nTFTRACKok=0;
        for(size_t i=0; i<ALCTs.size(); i++) if (ALCTs[i].deltaOk) nALCTok++;
        for(size_t i=0; i<CLCTs.size(); i++) if (CLCTs[i].deltaOk) nCLCTok++;
        for(size_t i=0; i<LCTs.size(); i++)  if (LCTs[i].deltaOk)  nLCTok++;
        for(size_t i=0; i<MPLCTs.size(); i++) if (MPLCTs[i].deltaOk) nMPLCTok++;
        int nok=0;
        for(size_t i=0; i<TFTRACKs.size(); i++) {
            for (size_t s=0; s<TFTRACKs[i].mplcts.size(); s++) if (TFTRACKs[i].mplcts[s]->deltaOk) nok++;
            if (nok>1) nTFTRACKok++;
        }
        std::cout<<"                 nALCTok="<<nALCTok<<"  nCLCTok="<<nCLCTok<<"  nLCTok="<<nLCTok<<"  nMPLCTok="<<nMPLCTok
            <<"  nTFTRACKok="<<nTFTRACKok<<std::endl;
        std::cout<<"                 eta, phi at ";
        for (size_t i=0; i<4; i++){
            math::XYZVectorD v = vAtStation(i);
            int st=i;
            if (i==0) st=11;
            std::cout<<"  ME"<<st<<": "<<v.eta()<<","<<v.phi();
        }
        std::cout<<std::endl;
        int key_st = keyStation();
        double dr_smart = deltaRSmart(strk->momentum().eta() , strk->momentum().phi());
        std::cout<<"                 DR to initial dir at station "<<key_st<<" is "<<dr_smart<<std::endl;
    }

    if (psimh) 
    {
        std::cout<<"****** SimTrack hits: total="<< nSimHits()<<" (is mu only ="<<muOnly<<"), in "<<hitsMapChamber.size()<<" chambers, in "<<hitsMapLayer.size()<<" detector IDs"<<std::endl;

        //self check 
        unsigned ntot=0;
        std::map<int, std::vector<PSimHit> >::const_iterator mapItr = hitsMapChamber.begin();
        for (; mapItr != hitsMapChamber.end(); mapItr++) 
        {
            unsigned nltot=0;
            std::map<int, std::vector<PSimHit> >::const_iterator lmapItr = hitsMapLayer.begin();
            for (; lmapItr != hitsMapLayer.end(); lmapItr++) 
            {
                CSCDetId lId(lmapItr->first);
                if (mapItr->first == (int)lId.chamberId().rawId()) nltot += hitsMapLayer[lmapItr->first].size();
            }
            if ( nltot != hitsMapChamber[mapItr->first].size() )
                std::cout<<" SELF CHACK ALARM!!! : chamber "<<mapItr->first<<" sum of hits in layers = "<<nltot<<" != # of hits in chamber "<<hitsMapChamber[mapItr->first].size()<<std::endl;
            ntot += nltot;
        }
        if (ntot != simHits.size()) 
            std::cout<<" SELF CHACK ALARM!!! : ntot hits in chambers = "<<ntot<<"!= simHits.size()"<<std::endl;


        std::vector<int> chIds = chambersWithHits(0,0,1);
        for (size_t ch = 0; ch < chIds.size(); ch++) {
            CSCDetId chid(chIds[ch]);
            std::pair<int,int> ws = wireGroupAndStripInChamber(chIds[ch]);
            std::cout<<"  chamber "<<chIds[ch]<<"   "<<chid<<"    #layers with hits = "<<numberOfLayersWithHitsInChamber(chIds[ch])<<"  w="<<ws.first<<"  s="<<ws.second<<std::endl;
            std::vector<PSimHit> chHits;
            if(DETAILED_HIT_LAYERS) chHits = chamberHits(chIds[ch]);
            for (unsigned i = 0; i < chHits.size(); i++) 
            {
                CSCDetId hid(chHits[i].detUnitId());
                std::cout<<"    L:"<<hid.layer()<<" "<<chHits[i]<<" "<<hid<<"  "<<chHits[i].momentumAtEntry()
                    <<" "<<chHits[i].energyLoss()<<" "<<chHits[i].particleType()<<" "<<chHits[i].trackId()<<std::endl;
            }
        }
        //    for (unsigned j=0; j<simHits.size(); j++) {
        //      CSCDetId hid(simHits[j].detUnitId());
        //      std::cout<<"    "<<simHits[j]<<" "<<hid<<"  "<<simHits[j].momentumAtEntry()
        //	  <<" "<<simHits[j].energyLoss()<<" "<<simHits[j].particleType()<<" "<<simHits[j].trackId()<<std::endl;
        //    }
    }

    if (palct) 
    {
        std::vector<int> chs = chambersWithALCTs();
        std::cout<<"****** match ALCTs: total="<< ALCTs.size()<<" in "<<chs.size()<<" chambers"<<std::endl;
        for (size_t c=0; c<chs.size(); c++)
        {
            std::vector<int> bxs = bxsWithALCTs( chs[c] );
            CSCDetId id(chs[c]);
            std::cout<<" ***** chamber "<<chs[c]<<"  "<<id<<"  has "<<bxs.size()<<" ALCT bxs"<<std::endl;
            for (size_t b=0; b<bxs.size(); b++)
            {
                std::vector<ALCT> stubs = chamberALCTsInBx( chs[c], bxs[b] );
                std::cout<<"   *** bx "<<bxs[b]<<" has "<<stubs.size()<<" ALCTs"<<std::endl;
                for (size_t i=0; i<stubs.size(); i++)
                {
                    std::cout<<"     * ALCT: "<<*(stubs[i].trgdigi)<<std::endl;
                    std::cout<<"       inReadOut="<<stubs[i].inReadOut()<<"  eta="<<stubs[i].eta<<"  deltaWire="<<stubs[i].deltaWire<<" deltaOk="<<stubs[i].deltaOk<<std::endl;
                    std::cout<<"       matched simhits to ALCT n="<<stubs[i].simHits.size()<<" nHitsShared="<<stubs[i].nHitsShared<<std::endl;
                    if (psimh) for (unsigned h=0; h<stubs[i].simHits.size();h++) 
                        std::cout<<"     "<<(stubs[i].simHits)[h]<<" "<<(stubs[i].simHits)[h].exitPoint()
                            <<"  "<<(stubs[i].simHits)[h].momentumAtEntry()<<" "<<(stubs[i].simHits)[h].energyLoss()
                            <<" "<<(stubs[i].simHits)[h].particleType()<<" "<<(stubs[i].simHits)[h].trackId()<<std::endl;
                }
            }
        }
    }

    if (pclct) 
    {
        std::vector<int> chs = chambersWithCLCTs();
        std::cout<<"****** match CLCTs: total="<< CLCTs.size()<<" in "<<chs.size()<<" chambers"<<std::endl;
        for (size_t c=0; c<chs.size(); c++)
        {
            std::vector<int> bxs = bxsWithCLCTs( chs[c] );
            CSCDetId id(chs[c]);
            std::cout<<" ***** chamber "<<chs[c]<<"  "<<id<<"  has "<<bxs.size()<<" CLCT bxs"<<std::endl;
            for (size_t b=0; b<bxs.size(); b++)
            {
                std::vector<CLCT> stubs = chamberCLCTsInBx( chs[c], bxs[b] );
                std::cout<<"   *** bx "<<bxs[b]<<" has "<<stubs.size()<<" CLCTs"<<std::endl;
                for (size_t i=0; i<stubs.size(); i++)
                {
                    std::cout<<"     * CLCT: "<<*(stubs[i].trgdigi)<<std::endl;
                    std::cout<<"       inReadOut="<<stubs[i].inReadOut()<<"  phi="<<stubs[i].phi<<"  deltaStrip="<<stubs[i].deltaStrip<<" deltaOk="<<stubs[i].deltaOk<<std::endl;
                    std::cout<<"       matched simhits to CLCT n="<<stubs[i].simHits.size()<<" nHitsShared="<<stubs[i].nHitsShared<<std::endl;
                    if (psimh) for (unsigned h=0; h<stubs[i].simHits.size();h++) 
                        std::cout<<"     "<<(stubs[i].simHits)[h]<<" "<<(stubs[i].simHits)[h].exitPoint()
                            <<"  "<<(stubs[i].simHits)[h].momentumAtEntry()<<" "<<(stubs[i].simHits)[h].energyLoss()
                            <<" "<<(stubs[i].simHits)[h].particleType()<<" "<<(stubs[i].simHits)[h].trackId()<<std::endl;
                }
            }
        }
    }

    if (plct)
    {
        std::vector<int> chs = chambersWithLCTs();
        std::cout<<"****** match LCTs: total="<< LCTs.size()<<" in "<<chs.size()<<" chambers"<<std::endl;
        for (size_t c=0; c<chs.size(); c++)
        {
            std::vector<int> bxs = bxsWithLCTs( chs[c] );
            CSCDetId id(chs[c]);
            std::cout<<" ***** chamber "<<chs[c]<<"  "<<id<<"  has "<<bxs.size()<<" LCT bxs"<<std::endl;
            for (size_t b=0; b<bxs.size(); b++)
            {
                std::vector<LCT> stubs = chamberLCTsInBx( chs[c], bxs[b] );
                std::cout<<"   *** bx "<<bxs[b]<<" has "<<stubs.size()<<" LCTs"<<std::endl;
                for (size_t i=0; i<stubs.size(); i++)
                {
                    bool matchALCT = (stubs[i].alct != 0), matchCLCT = (stubs[i].clct != 0);
                    std::cout<<"     * LCT: "<<*(stubs[i].trgdigi);
                    std::cout<<"         is ghost="<<stubs[i].ghost<<"  inReadOut="<<stubs[i].inReadOut()
                        <<"  found assiciated ALCT="<< matchALCT <<" CLCT="<< matchCLCT <<std::endl;
                    if (matchALCT && matchCLCT)
                        std::cout<<"         BX(A)-BX(C)="<<stubs[i].alct->getBX() - stubs[i].clct->getBX()
                            <<"  deltaWire="<<stubs[i].alct->deltaWire<<"  deltaStrip="<<stubs[i].clct->deltaStrip
                            <<"  deltaOk="<<stubs[i].deltaOk<<"="<<stubs[i].alct->deltaOk<<"&"<<stubs[i].clct->deltaOk<<std::endl;

                }
            }
        }
    }

    if (pmplct)
    {
        std::vector<int> chs = chambersWithMPLCTs();
        std::cout<<"****** match MPLCTs: total="<< MPLCTs.size()<<" in "<<chs.size()<<" chambers"<<std::endl;
        for (size_t c=0; c<chs.size(); c++)
        {
            std::vector<int> bxs = bxsWithMPLCTs( chs[c] );
            CSCDetId id(chs[c]);
            std::cout<<" ***** chamber "<<chs[c]<<"  "<<id<<"  has "<<bxs.size()<<" MPLCT bxs"<<std::endl;
            for (size_t b=0; b<bxs.size(); b++)
            {
                std::vector<MPLCT> stubs = chamberMPLCTsInBx( chs[c], bxs[b] );
                std::cout<<"   *** bx "<<bxs[b]<<" has "<<stubs.size()<<" MPLCTs"<<std::endl;
                for (size_t i=0; i<stubs.size(); i++)
                {
                    bool matchLCT = (stubs[i].lct != 0);
                    std::cout<<"     * MPLCT: "<<*(stubs[i].trgdigi);
                    std::cout<<"         is ghost="<<stubs[i].ghost<<"  inReadOut="<<stubs[i].inReadOut()
                        <<"  found associated LCT="<<matchLCT<<std::endl;
                    if (matchLCT) {
                        if (stubs[i].lct->alct != 0 && stubs[i].lct->clct != 0) 
                            std::cout<<"         BX(A)-BX(C)="<<stubs[i].lct->alct->getBX() - stubs[i].lct->clct->getBX()
                                <<"  deltaWire="<<stubs[i].lct->alct->deltaWire<<"  deltaStrip="<<stubs[i].lct->clct->deltaStrip
                                <<"  deltaOk="<<stubs[i].deltaOk<<"="<<stubs[i].lct->alct->deltaOk<<"&"<<stubs[i].lct->clct->deltaOk<<std::endl;
                    }
                    else std::cout<<"       deltaOk="<<stubs[i].deltaOk<<std::endl;
                }
            }
        }
    }

    if (ptftrack){}

    if (ptfcand)
    {
        std::cout<<"--- match TFCANDs: total="<< TFCANDs.size()<<std::endl;
        for (size_t i=0; i<TFCANDs.size(); i++)
        {
            char tfi[4];
            sprintf(tfi," TFTrack %lu",i);
            if (TFCANDs[i].tftrack) TFCANDs[i].tftrack->print(tfi);
            else std::cout<<"Strange: tfcand "<<i<<" has no tftrack!!!"<<std::endl;
        }
    }

    std::cout<<"####### END MATCH PRINT #######"<<std::endl;
}


//_____________________________________________________________________________
/*
 * Get the best ALCT for a particular chamber; best is defined as the ALCT with 
 * the least difference in wire group number
 */
    MatchCSCMuL1::ALCT * 
MatchCSCMuL1::bestALCT(CSCDetId id, bool readout)
{
    if (ALCTs.size()==0) return NULL;
    //double minDY=9999.;
    int minDW=9999;
    unsigned minN=99;
    for (unsigned i=0; i<ALCTs.size();i++) 
        if (id.chamberId().rawId() == ALCTs[i].id.chamberId().rawId())
            if (!readout || ALCTs[i].inReadOut())
                //if (fabs(ALCTs[i].deltaY)<minDY) { minDY = fabs(ALCTs[i].deltaY); minN=i;}
                if (abs(ALCTs[i].deltaWire)<minDW) { minDW = abs(ALCTs[i].deltaWire); minN=i;}
    if (minN==99) return NULL;
    return &(ALCTs[minN]);
}


//_____________________________________________________________________________
/*
 * Get the best CLCT for a particular chamber; best is defined as the CLCT with 
 * the least difference in strip number
 */
    MatchCSCMuL1::CLCT * 
MatchCSCMuL1::bestCLCT(CSCDetId id, bool readout)
{
    if (CLCTs.size()==0) return NULL;
    //double minDY=9999.;
    int minDS=9999;
    unsigned minN=99;
    for (unsigned i=0; i<CLCTs.size();i++) 
        if (id.chamberId().rawId() == CLCTs[i].id.chamberId().rawId())
            if (!readout || CLCTs[i].inReadOut())
                //if (fabs(CLCTs[i].deltaY)<minDY) { minDY = fabs(CLCTs[i].deltaY); minN=i;}
                if (abs(CLCTs[i].deltaStrip)<minDS) { minDS = abs(CLCTs[i].deltaStrip); minN=i;}
    if (minN==99) return NULL;
    return &(CLCTs[minN]);
}


//_____________________________________________________________________________
/*
 * Return the best TrackFinder track from a collection of TFTracks
 * Option to sort the TFTracks according to pt
 */
    MatchCSCMuL1::TFTRACK * 
MatchCSCMuL1::bestTFTRACK(std::vector< TFTRACK > & tracks, bool sortPtFirst)
{
    if (tracks.size()==0) return NULL;

    // determine max # of matched stubs in the TFTrack collection
    int maxNMatchedMPC = 0;
    for (unsigned i=0; i<tracks.size(); i++) 
    {
        int nst=0;
        for (size_t s=0; s<tracks[i].ids.size(); s++) 
            if (tracks[i].mplcts[s]->deltaOk) nst++;
        if (nst>maxNMatchedMPC) maxNMatchedMPC = nst;
    }
    // collect tracks with max # of matched stubs
    std::vector<unsigned> bestMatchedTracks;
    for (unsigned i=0; i<tracks.size(); i++) 
    {
        int nst=0;
        for (size_t s=0; s<tracks[i].ids.size(); s++) 
            if (tracks[i].mplcts[s]->deltaOk) nst++;
        if (nst==maxNMatchedMPC) bestMatchedTracks.push_back(i);
    }

    // already found the best TFTrack
    if (bestMatchedTracks.size()==1) return &(tracks[bestMatchedTracks[0]]);

    // case when you have more than 1 best TFTrack
    // first sort by quality
    double qBase  = 1000000.;
    // then sort by Pt inside the cone (if sortPtFirst), then sort by DR
    double ptBase = 0.;
    if (sortPtFirst) ptBase = 1000.;
    unsigned maxI = 99;
    double maxRank = -999999.;
    for (unsigned i=0; i<tracks.size(); i++) 
    {
        if (bestMatchedTracks.size()) {
            bool gotit=0;
            for (unsigned m=0;m<bestMatchedTracks.size();m++) if (bestMatchedTracks[m]==i) gotit=1;
            if (!gotit) continue;
        }
        double rr = qBase*tracks[i].q_packed + ptBase*tracks[i].pt_packed + 1./(0.01 + tracks[i].dr);
        if (rr > maxRank) { maxRank = rr; maxI = i;}
    }
    if (maxI==99) return NULL;
    return &(tracks[maxI]);
}


//_____________________________________________________________________________
/*
 * Return the best TrackFinder candidate from a collection of TFCands
 * Option to sort the TFCands according to pt
 */
    MatchCSCMuL1::TFCAND * 
MatchCSCMuL1::bestTFCAND(std::vector< TFCAND > & cands, bool sortPtFirst)
{
    if (cands.size()==0) return NULL;

    // determine max # of matched stubs
    int maxNMatchedMPC = 0;
    for (unsigned i=0; i<cands.size(); i++) 
    {
        int nst=0;
        if (cands[i].tftrack==0) continue;
        for (size_t s=0; s<cands[i].tftrack->ids.size(); s++) 
            if (cands[i].tftrack->mplcts[s]->deltaOk) nst++;
        if (nst>maxNMatchedMPC) maxNMatchedMPC = nst;
    }

    // collect tracks with max # of matched stubs
    std::vector<unsigned> bestMatchedTracks;
    if (maxNMatchedMPC>0) {
        for (unsigned i=0; i<cands.size(); i++) 
        {
            int nst=0;
            if (cands[i].tftrack==0) continue;
            for (size_t s=0; s<cands[i].tftrack->ids.size(); s++) 
                if (cands[i].tftrack->mplcts[s]->deltaOk) nst++;
            if (nst==maxNMatchedMPC) bestMatchedTracks.push_back(i);
        }
        if (bestMatchedTracks.size()==1) return &(cands[bestMatchedTracks[0]]);
    }

    // first sort by quality
    double qBase  = 1000000.;
    // then sort by Pt inside the cone (if sortPtFirst), then sort by DR
    double ptBase = 0.;
    if (sortPtFirst) ptBase = 1000.;
    unsigned maxI = 99;
    double maxRank = -999999.;
    for (unsigned i=0; i<cands.size(); i++) 
    {
        if (bestMatchedTracks.size()) {
            bool gotit=0;
            for (unsigned m=0;m<bestMatchedTracks.size();m++) if (bestMatchedTracks[m]==i) gotit=1;
            if (!gotit) continue;
        }
        // quality criterium you apply to get the best TFCand
        double rr = qBase*cands[i].l1cand->quality_packed() + ptBase*cands[i].l1cand->pt_packed() + 1./(0.01 + cands[i].dr);
        if (rr > maxRank) { maxRank = rr; maxI = i;}
    }
    if (maxI==99) return NULL;
    return &(cands[maxI]);
}


//_____________________________________________________________________________
/*
 * Return the best GMT Regional Candidate from a collection of GMT Regional Candidates
 * Option to sort the GMT Regional Candidates to pt
 */
    MatchCSCMuL1::GMTREGCAND * 
MatchCSCMuL1::bestGMTREGCAND(std::vector< GMTREGCAND > & cands, bool sortPtFirst)
{
    // first sort by Pt inside the cone (if sortPtFirst), then sort by DR
    if (cands.size()==0) return NULL;
    double ptBase = 0.;
    if (sortPtFirst) ptBase = 1000.;
    unsigned maxI = 99;
    double maxRank = -999999.;
    for (unsigned i=0; i<cands.size(); i++) 
    {
        // quality criterium to sort the GMT Regional candidates
        double rr = ptBase*cands[i].pt + 1./(0.01 + cands[i].dr);
        if (rr > maxRank) { maxRank = rr; maxI = i;}
    }
    if (maxI==99) return NULL;
    return &(cands[maxI]);
}


//_____________________________________________________________________________
/*
 * Return the best GMT Candidate from a collection of GMT Candidates
 * Option to sort the GMT Candidates to pt
 */
    MatchCSCMuL1::GMTCAND * 
MatchCSCMuL1::bestGMTCAND(std::vector< GMTCAND > & cands, bool sortPtFirst)
{
    // first sort by Pt inside the cone (if sortPtFirst), then sort by DR
    if (cands.size()==0) return NULL;
    double ptBase = 0.;
    if (sortPtFirst) ptBase = 1000.;
    unsigned maxI = 99;
    double maxRank = -999999.;
    for (unsigned i=0; i<cands.size(); i++) 
    {
        // quality criterium to sort the GMT candidates
        double rr = ptBase*cands[i].pt + 1./(0.01 + cands[i].dr);
        if (rr > maxRank) { maxRank = rr; maxI = i;}
    }
    if (maxI==99) return NULL;
    return &(cands[maxI]);
}



MatchCSCMuL1::ALCT::ALCT():match(0),trgdigi(0) {}
MatchCSCMuL1::ALCT::ALCT(MatchCSCMuL1 *m):match(m),trgdigi(0) {}

MatchCSCMuL1::CLCT::CLCT():match(0),trgdigi(0) {}
MatchCSCMuL1::CLCT::CLCT(MatchCSCMuL1 *m):match(m),trgdigi(0) {}

MatchCSCMuL1::LCT::LCT():match(0),trgdigi(0) {}
MatchCSCMuL1::LCT::LCT(MatchCSCMuL1 *m):match(m),trgdigi(0) {}

MatchCSCMuL1::MPLCT::MPLCT():match(0),trgdigi(0) {}
MatchCSCMuL1::MPLCT::MPLCT(MatchCSCMuL1 *m):match(m),trgdigi(0) {}

MatchCSCMuL1::TFTRACK::TFTRACK():match(0),l1trk(0), deltaOk1(0), deltaOk2(0), deltaOkME1(0), debug(0) {}
MatchCSCMuL1::TFTRACK::TFTRACK(MatchCSCMuL1 *m):match(m),l1trk(0), deltaOk1(0), deltaOk2(0), deltaOkME1(0), debug(0) {}

MatchCSCMuL1::TFCAND::TFCAND():match(0),l1cand(0) {}
MatchCSCMuL1::TFCAND::TFCAND(MatchCSCMuL1 *m):match(m),l1cand(0) {}


//_____________________________________________________________________________
    bool 
MatchCSCMuL1::ALCT::inReadOut()
{
    return getBX()>=match->minBxALCT && getBX()<=match->maxBxALCT;
}


//_____________________________________________________________________________
    bool 
MatchCSCMuL1::CLCT::inReadOut()
{
    return getBX()>=match->minBxCLCT && getBX()<=match->maxBxCLCT;
}


//_____________________________________________________________________________
    bool 
MatchCSCMuL1::LCT::inReadOut()
{
    return getBX()>=match->minBxLCT && getBX()<=match->maxBxLCT;
}


//_____________________________________________________________________________
    bool 
MatchCSCMuL1::MPLCT::inReadOut()
{
    return getBX()>=match->minBxMPLCT && getBX()<=match->maxBxMPLCT;
}


//_____________________________________________________________________________
    void 
MatchCSCMuL1::TFTRACK::init(const csc::L1Track *t, CSCTFPtLUT* ptLUT,
        edm::ESHandle< L1MuTriggerScales > &muScales,
        edm::ESHandle< L1MuTriggerPtScale > &muPtScale)
{
    l1trk = t;

    unsigned gbl_phi = t->localPhi() + ((t->sector() - 1)*24) + 6; //for now, convert using this. LUT in the future
    if(gbl_phi > 143) gbl_phi -= 143;
    phi_packed = gbl_phi & 0xff;

    unsigned eta_sign = (t->endcap() == 1 ? 0 : 1);
    int gbl_eta = t->eta_packed() | eta_sign << (L1MuRegionalCand::ETA_LENGTH - 1);
    eta_packed  = gbl_eta & 0x3f;

    unsigned rank = t->rank(), gpt = 0, quality = 0;
    //if (rank != 0 ) {
    //  quality = rank >> L1MuRegionalCand::PT_LENGTH;
    //  gpt = rank & ( (1<<L1MuRegionalCand::PT_LENGTH) - 1);
    //}
    csc::L1Track::decodeRank(rank, gpt, quality);
    q_packed = quality & 0x3;
    pt_packed = gpt & 0x1f;

    //pt = muPtScale->getPtScale()->getLowEdge(pt_packed) + 1.e-6;
    eta = muScales->getRegionalEtaScale(2)->getCenter(t->eta_packed());
    phi = normalizedPhi( muScales->getPhiScale()->getLowEdge(phi_packed));

    //Pt needs some more workaround since it is not in the unpacked data
    //  PtAddress gives an handle on other parameters
    ptadd thePtAddress(t->ptLUTAddress());
    ptdat thePtData  = ptLUT->Pt(thePtAddress);
    // front or rear bit? 
    unsigned trPtBit = (thePtData.rear_rank&0x1f);
    if (thePtAddress.track_fr) trPtBit = (thePtData.front_rank&0x1f);
    // convert the Pt in human readable values (GeV/c)
    pt  = muPtScale->getPtScale()->getLowEdge(trPtBit); 

    //if (trPtBit!=pt_packed) std::cout<<" trPtBit!=pt_packed: "<<trPtBit<<"!="<<pt_packed<<"  pt="<<pt<<" eta="<<eta<<std::endl;

    bool sc_debug = 0;
    if (sc_debug && deltaOk2){
        double stpt=-99., steta=-99., stphi=-99.;
        if (match){
            stpt = sqrt(match->strk->momentum().perp2());
            steta = match->strk->momentum().eta();
            stphi = normalizedPhi( match->strk->momentum().phi() );
        }
        //debug = 1;

        double my_phi = normalizedPhi( phi_packed*0.043633231299858237 + 0.0218 ); // M_PI*2.5/180 = 0.0436332312998582370
        double my_eta = 0.05 * eta_packed + 0.925; //  0.9+0.025 = 0.925
        //double my_pt = ptscale[pt_packed];
        //if (fabs(pt - my_pt)>0.005) std::cout<<"scales pt diff: my "<<my_pt<<"  sc: pt "<<pt<<"  eta "<<eta<<" phi "<<phi<<"  mc: pt "<<stpt<<"  eta "<<steta<<" phi "<<stphi<<std::endl;
        if (fabs(eta - my_eta)>0.005) std::cout<<"scales eta diff: my "<<my_eta<<" sc "<<eta<<"  mc: pt "<<stpt<<"  eta "<<steta<<" phi "<<stphi<<std::endl;
        if (fabs(deltaPhi(phi,my_phi))>0.03) std::cout<<"scales phi diff: my "<<my_phi<<" sc "<<phi<<"  mc: pt "<<stpt<<"  eta "<<steta<<" phi "<<stphi<<std::endl;

        double old_pt = muPtScale->getPtScale()->getLowEdge(pt_packed) + 1.e-6;
        if (fabs(pt - old_pt)>0.005) { debug = 1;std::cout<<"lut pt diff: old "<<old_pt<<" lut "<<pt<<"  eta "<<eta<<" phi "<<phi<<"   mc: pt "<<stpt<<"  eta "<<steta<<" phi "<<stphi<<std::endl;}
        double lcl_phi = normalizedPhi( fmod( muScales->getPhiScale()->getLowEdge(t->localPhi()) + 
                    (t->sector()-1)*M_PI/3. + //sector 1 starts at 15 degrees 
                    M_PI/12. , 2.*M_PI) );
        if (fabs(deltaPhi(phi,lcl_phi))>0.03) std::cout<<"lcl phi diff: lcl "<<lcl_phi<<" sc "<<phi<<"  mc: pt "<<stpt<<"  eta "<<steta<<" phi "<<stphi<<std::endl;
    }
}


//_____________________________________________________________________________
/*
 * Has this TFTrack a stub in station 1,2,3,4 or in the muon barrel?
 */
    bool 
MatchCSCMuL1::TFTRACK::hasStub(int st)
{
    if(st==0 && l1trk->mb1ID() > 0) return true;
    if(st==1 && l1trk->me1ID() > 0) return true;
    if(st==2 && l1trk->me2ID() > 0) return true;
    if(st==3 && l1trk->me3ID() > 0) return true;
    if(st==4 && l1trk->me4ID() > 0) return true;
    return false;
}


//_____________________________________________________________________________
/*
 * Has this TFTrack a good CSC stub?
 */
    bool 
MatchCSCMuL1::TFTRACK::hasStubCSCOk(int st)
{
    if (!hasStub(st)) return false;
    bool cscok = 0;
    for (size_t s=0; s<ids.size(); s++)
        if (ids[s].station() == st && mplcts[s]->deltaOk) { cscok = 1; break; }
    if (cscok) return true;
    return false;
}


//_____________________________________________________________________________
/*
 * How many stubs does this TFTrack has?
 */
    unsigned int 
MatchCSCMuL1::TFTRACK::nStubs(bool mb1, bool me1, bool me2, bool me3, bool me4)
{
    return (mb1 & hasStub(0)) + (me1 & hasStub(1)) + (me2 & hasStub(2)) + (me3 & hasStub(3)) + (me4 & hasStub(4));
}


//_____________________________________________________________________________
/*
 * How many good CSC stubs does this TFTrack has?
 */
    unsigned int 
MatchCSCMuL1::TFTRACK::nStubsCSCOk(bool mb1, bool me1, bool me2, bool me3, bool me4)
{
    return (me1 & hasStubCSCOk(1)) + (me2 & hasStubCSCOk(2)) + (me3 & hasStubCSCOk(3)) + (me4 & hasStubCSCOk(4));
}


//_____________________________________________________________________________
/*
 * Does this track has the necessary number of stubs?
 */
    bool 
MatchCSCMuL1::TFTRACK::passStubsMatch(int minLowHStubs, int minMidHStubs, int minHighHStubs)
{
    double steta = match->strk->momentum().eta();
    int nstubs = nStubs(1,1,1,1,1);
    int nstubsok = nStubsCSCOk(1,1,1,1);
    if (fabs(steta) <= 1.2)      return nstubsok >=1 && nstubs >= minLowHStubs;
    else if (fabs(steta) <= 2.1) return nstubsok >=2 && nstubs >= minMidHStubs;
    else                         return nstubsok >=2 && nstubs >= minHighHStubs;
}


//_____________________________________________________________________________
    void 
MatchCSCMuL1::TFTRACK::print(const char msg[300])
{
    std::cout<<"#### TFTRACK PRINT: "<<msg<<" #####"<<std::endl;
    //std::cout<<"## L1MuRegionalCand print: ";
    //l1trk->print();
    //std::cout<<"\n## L1Track Print: ";
    //l1trk->Print();
    //std::cout<<"## TFTRACK:  
    std::cout<<"\tpt_packed: "<<pt_packed<<"  eta_packed: " << eta_packed<<"  phi_packed: " << phi_packed<<"  q_packed: "<< q_packed<<"  bx: "<<l1trk->bx()<<std::endl;
    std::cout<<"\tpt: "<<pt<<"  eta: "<<eta<<"  phi: "<<phi<<"  sector: "<<l1trk->sector()<<"  dr: "<<dr<<"   ok1: "<<deltaOk1<<"  ok2: "<<deltaOk2<<"  okME1: "<<deltaOkME1<<std::endl;
    std::cout<<"\tMB1 ME1 ME2 ME3 ME4 = "<<l1trk->mb1ID()<<" "<<l1trk->me1ID()<<" "<<l1trk->me2ID()<<" "<<l1trk->me3ID()<<" "<<l1trk->me4ID()
        <<" ("<<hasStub(0)<<" "<<hasStub(1)<<" "<<hasStub(2)<<" "<<hasStub(3)<<" "<<hasStub(4)<<")  "
        <<" ("<<hasStubCSCOk(1)<<" "<<hasStubCSCOk(2)<<" "<<hasStubCSCOk(3)<<" "<<hasStubCSCOk(4)<<")"<<std::endl;
    std::cout<<"\tptAddress: 0x"<<std::hex<<l1trk->ptLUTAddress()<<std::dec<<"  mode: "<<mode()<<"  sign: "<<sign()<<"  dphi12: "<<dPhi12()<<"  dphi23: "<<dPhi23()<<std::endl;
    std::cout<<"\thas "<<trgdigis.size()<<" stubs in ";
    for (size_t s=0; s<trgids.size(); s++) 
        std::cout<<trgids[s]<<" w:"<<trgdigis[s]->getKeyWG()<<" s:"<<trgdigis[s]->getStrip()/2 + 1<<" p:"<<trgdigis[s]->getPattern()<<" bx:"<<trgdigis[s]->getBX()<<"; ";
    std::cout<<std::endl;
    std::cout<<"\tstub_etaphis:";
    for (size_t s=0; s<trgids.size(); s++)
        std::cout<<"  "<<trgetaphis[s].first<<" "<<trgetaphis[s].second;
    std::cout<<std::endl;
    std::cout<<"\tstub_petaphis:";
    for (size_t s=0; s<trgstubs.size(); s++)
        std::cout<<"  "<<trgstubs[s].etaPacked()<<" "<<trgstubs[s].phiPacked();
    std::cout<<std::endl;
    std::cout<<"\thas "<<mplcts.size()<<" associated MPCs in ";
    for (size_t s=0; s<ids.size(); s++) 
        std::cout<<ids[s]<<" w:"<<mplcts[s]->trgdigi->getKeyWG()<<" s:"<<mplcts[s]->trgdigi->getStrip()/2 + 1<<" Ok="<<mplcts[s]->deltaOk<<"; ";
    std::cout<<std::endl;
    std::cout<<"\tMPCs meEtap and mePhip: ";
    for (size_t s=0; s<ids.size(); s++) std::cout<<mplcts[s]->meEtap<<", "<<mplcts[s]->mePhip<<";  ";
    std::cout<<std::endl;
    std::cout<<"#### TFTRACK END PRINT #####"<<std::endl;
}


//_____________________________________________________________________________
    void
MatchCSCMuL1::TFCAND::init(const L1MuRegionalCand *t, CSCTFPtLUT* ptLUT,
        edm::ESHandle< L1MuTriggerScales > &muScales,
        edm::ESHandle< L1MuTriggerPtScale > &muPtScale)
{
    l1cand = t;

    pt = muPtScale->getPtScale()->getLowEdge(t->pt_packed()) + 1.e-6;
    eta = muScales->getRegionalEtaScale(2)->getCenter(t->eta_packed());
    phi = normalizedPhi( muScales->getPhiScale()->getLowEdge(t->phi_packed()));
    nTFStubs = -1;
    /*
    //Pt needs some more workaround since it is not in the unpacked data
    //  PtAddress gives an handle on other parameters
    ptadd thePtAddress(t->ptLUTAddress());
    ptdat thePtData  = ptLUT->Pt(thePtAddress);
    // front or rear bit? 
    unsigned trPtBit = (thePtData.rear_rank&0x1f);
    if (thePtAddress.track_fr) trPtBit = (thePtData.front_rank&0x1f);
    // convert the Pt in human readable values (GeV/c)
    pt  = muPtScale->getPtScale()->getLowEdge(trPtBit); 
     */
    bool sc_debug = 1;
    if (sc_debug){
        double my_phi = normalizedPhi( t->phi_packed()*0.043633231299858237 + 0.0218 ); // M_PI*2.5/180 = 0.0436332312998582370
        double sign_eta = ( (t->eta_packed() & 0x20) == 0) ? 1.:-1;
        double my_eta = sign_eta*(0.05 * (t->eta_packed() & 0x1F) + 0.925); //  0.9+0.025 = 0.925
        double my_pt = ptscale[t->pt_packed()];
        if (fabs(pt - my_pt)>0.005) std::cout<<"tfcand scales pt diff: my "<<my_pt<<" sc "<<pt<<std::endl;
        if (fabs(eta - my_eta)>0.005) std::cout<<"tfcand scales eta diff: my "<<my_eta<<" sc "<<eta<<std::endl;
        if (fabs(deltaPhi(phi,my_phi))>0.03) std::cout<<"tfcand scales phi diff: my "<<my_phi<<" sc "<<phi<<std::endl;
    }  
}


//_____________________________________________________________________________
    void 
MatchCSCMuL1::GMTREGCAND::print(const char msg[300])
{
    std::string sys="Mu";
    if (l1reg->type_idx()==2) sys = "CSC";
    if (l1reg->type_idx()==3) sys = "RPCf";
    std::cout<<"#### GMTREGCAND ("<<sys<<") PRINT: "<<msg<<" #####"<<std::endl;
    //l1reg->print();
    std::cout<<" bx="<<l1reg->bx()<<" values: pt="<<pt<<" eta="<<eta<<" phi="<<phi<<" packed: pt="<<l1reg->pt_packed()<<" eta="<<eta_packed<<" phi="<<phi_packed<<"  q="<<l1reg->quality()<<"  ch="<<l1reg->chargeValue()<<" chOk="<<l1reg->chargeValid()<<std::endl;
    if (tfcand!=NULL) std::cout<<"has tfcand with "<<ids.size()<<" stubs"<<std::endl;
    std::cout<<"#### GMTREGCAND END PRINT #####"<<std::endl;
}


//_____________________________________________________________________________
void MatchCSCMuL1::GMTREGCAND::init(const L1MuRegionalCand *t,
        edm::ESHandle< L1MuTriggerScales > &muScales,
        edm::ESHandle< L1MuTriggerPtScale > &muPtScale)
{
    l1reg = t;

    pt = muPtScale->getPtScale()->getLowEdge(t->pt_packed()) + 1.e-6;
    eta = muScales->getRegionalEtaScale(t->type_idx())->getCenter(t->eta_packed());
    //std::cout<<"regetac"<<t->type_idx()<<"="<<eta<<std::endl;
    //std::cout<<"regetalo"<<t->type_idx()<<"="<<muScales->getRegionalEtaScale(t->type_idx())->getLowEdge(t->eta_packed() )<<std::endl;
    phi = normalizedPhi( muScales->getPhiScale()->getLowEdge(t->phi_packed()));
    nTFStubs = -1;

    bool sc_debug = 0;
    if (sc_debug){
        double my_phi = normalizedPhi( t->phi_packed()*0.043633231299858237 + 0.0218 ); // M_PI*2.5/180 = 0.0436332312998582370
        double sign_eta = ( (t->eta_packed() & 0x20) == 0) ? 1.:-1;
        double my_eta = sign_eta*(0.05 * (t->eta_packed() & 0x1F) + 0.925); //  0.9+0.025 = 0.925
        double my_pt = ptscale[t->pt_packed()];
        if (fabs(pt - my_pt)>0.005) std::cout<<"gmtreg scales pt diff: my "<<my_pt<<" sc "<<pt<<std::endl;
        if (fabs(eta - my_eta)>0.005) std::cout<<"gmtreg scales eta diff: my "<<my_eta<<" sc "<<eta<<std::endl;
        if (fabs(deltaPhi(phi,my_phi))>0.03) std::cout<<"gmtreg scales phi diff: my "<<my_phi<<" sc "<<phi<<std::endl;
    }  
}


//_____________________________________________________________________________
void MatchCSCMuL1::GMTCAND::init(const L1MuGMTExtendedCand *t,
        edm::ESHandle< L1MuTriggerScales > &muScales,
        edm::ESHandle< L1MuTriggerPtScale > &muPtScale)
{
    l1gmt = t;

    // keep x and y components non-zero and protect against roundoff.
    pt = muPtScale->getPtScale()->getLowEdge( t->ptIndex() ) + 1.e-6 ;
    eta = muScales->getGMTEtaScale()->getCenter( t->etaIndex() ) ;
    //std::cout<<"gmtetalo="<<muScales->getGMTEtaScale()->getLowEdge(t->etaIndex() )<<std::endl;
    //std::cout<<"gmtetac="<<eta<<std::endl;
    phi = normalizedPhi( muScales->getPhiScale()->getLowEdge( t->phiIndex() ) ) ;
    math::PtEtaPhiMLorentzVector p4( pt, eta, phi, MUON_MASS );
    pt = p4.pt();
    q = t->quality();
    rank = t->rank();
}

