/*
 * \file EcalEndcapSimHitsValidation.cc
 *
 * \author C.Rovelli
 *
*/

#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include "FWCore/Utilities/interface/Exception.h"
#include "Validation/EcalHits/interface/EcalEndcapSimHitsValidation.h"

using namespace cms;
using namespace edm;
using namespace std;

EcalEndcapSimHitsValidation::EcalEndcapSimHitsValidation(const edm::ParameterSet& ps):
  g4InfoLabel(ps.getParameter<std::string>("moduleLabelG4")),
  EEHitsCollection(ps.getParameter<std::string>("EEHitsCollection")),
  ValidationCollection(ps.getParameter<std::string>("ValidationCollection")){   

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
 
  if ( verbose_ ) {
    std::cout << " verbose switch is ON" << std::endl;
  } else {
    std::cout << " verbose switch is OFF" << std::endl;
  }

  // get hold of back-end interface
  dbe_ = 0;
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();           
  if ( dbe_ ) {
    if ( verbose_ ) { dbe_->setVerbose(1); } 
    else            { dbe_->setVerbose(0); }
  }
                                                                                                            
  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }
 

  meEEzpHits_             = 0;
  meEEzmHits_             = 0;
  meEEzpCrystals_         = 0;
  meEEzmCrystals_         = 0;
  meEEzpOccupancy_        = 0;
  meEEzmOccupancy_        = 0;
  meEELongitudinalShower_ = 0;
  meEEzpHitEnergy_        = 0;
  meEEzmHitEnergy_        = 0;

  meEEe1_  = 0;  
  meEEe4_  = 0;  
  meEEe9_  = 0;  
  meEEe16_ = 0; 
  meEEe25_ = 0; 

  meEEe1oe4_   = 0;  
  meEEe4oe9_   = 0;  
  meEEe9oe16_  = 0;
  meEEe1oe25_  = 0;  
  meEEe9oe25_  = 0; 
  meEEe16oe25_ = 0;

  Char_t histo[200];
 
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalSimHitsValidation");
  
    sprintf (histo, "EE+ hits multiplicity" ) ;
    meEEzpHits_ = dbe_->book1D(histo, histo, 50, 0., 5000.) ; 

    sprintf (histo, "EE- hits multiplicity" ) ;
    meEEzmHits_ = dbe_->book1D(histo, histo, 50, 0., 5000.) ; 

    sprintf (histo, "EE+ crystals multiplicity" ) ;
    meEEzpCrystals_ = dbe_->book1D(histo, histo, 50, 0., 300.) ; 

    sprintf (histo, "EE- crystals multiplicity" ) ;
    meEEzmCrystals_ = dbe_->book1D(histo, histo, 50, 0., 300.) ; 

    sprintf (histo, "EE+ occupancy" ) ;
    meEEzpOccupancy_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  
    sprintf (histo, "EE- occupancy" ) ;
    meEEzmOccupancy_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  
    sprintf (histo, "EE longitudinal shower profile" ) ;
    meEELongitudinalShower_ = dbe_->bookProfile(histo, histo, 26,0,26, 100, 0, 3000);

    sprintf (histo, "EE+ hits energy spectrum" );
    meEEzpHitEnergy_ = dbe_->book1D(histo, histo, 4000, 0., 400.);

    sprintf (histo, "EE- hits energy spectrum" );
    meEEzmHitEnergy_ = dbe_->book1D(histo, histo, 4000, 0., 400.);

    sprintf (histo, "EE E1" ) ;
    meEEe1_ = dbe_->book1D(histo, histo, 400, 0., 400.);

    sprintf (histo, "EE E4" ) ;
    meEEe4_ = dbe_->book1D(histo, histo, 400, 0., 400.);

    sprintf (histo, "EE E9" ) ;
    meEEe9_ = dbe_->book1D(histo, histo, 400, 0., 400.);

    sprintf (histo, "EE E16" ) ;
    meEEe16_ = dbe_->book1D(histo, histo, 400, 0., 400.);

    sprintf (histo, "EE E25" ) ;
    meEEe25_ = dbe_->book1D(histo, histo, 400, 0., 400.);

    sprintf (histo, "EE E1oE4" ) ;
    meEEe1oe4_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EE E4oE9" ) ;
    meEEe4oe9_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EE E9oE16" ) ;
    meEEe9oe16_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EE E1oE25" ) ;
    meEEe1oe25_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EE E9oE25" ) ;
    meEEe9oe25_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);

    sprintf (histo, "EE E16oE25" ) ;
    meEEe16oe25_ = dbe_->book1D(histo, histo, 100, 0.4, 1.1);
  }
}

EcalEndcapSimHitsValidation::~EcalEndcapSimHitsValidation(){

}

void EcalEndcapSimHitsValidation::beginJob(const edm::EventSetup& c){

}

void EcalEndcapSimHitsValidation::endJob(){

}

void EcalEndcapSimHitsValidation::analyze(const edm::Event& e, const edm::EventSetup& c){

  edm::LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
  
  edm::Handle<edm::PCaloHitContainer> EcalHitsEE;
  e.getByLabel(g4InfoLabel,EEHitsCollection,EcalHitsEE);

  edm::Handle<PEcalValidInfo> MyPEcalValidInfo;
  e.getByLabel(g4InfoLabel,ValidationCollection,MyPEcalValidInfo);

  std::vector<PCaloHit> theEECaloHits;
  theEECaloHits.insert(theEECaloHits.end(), EcalHitsEE->begin(), EcalHitsEE->end());
  
  std::map<unsigned int, std::vector<PCaloHit>,std::less<unsigned int> > CaloHitMap;
  
  double EEetzp_ = 0.;
  double EEetzm_ = 0.;
  
  double ee1  = 0.0;
  double ee4  = 0.0;
  double ee9  = 0.0;
  double ee16 = 0.0;
  double ee25 = 0.0;
  
  MapType eemap;
  MapType eemapzp;
  MapType eemapzm;
  uint32_t nEEzpHits = 0;
  uint32_t nEEzmHits = 0;
  
  for (std::vector<PCaloHit>::iterator isim = theEECaloHits.begin();
       isim != theEECaloHits.end(); ++isim){
    CaloHitMap[(*isim).id()].push_back((*isim));
    
    EEDetId eeid (isim->id()) ;
    
    LogDebug("HitInfo") 
      << " CaloHit " << isim->getName() << "\n" 
      << " DetID = "<<isim->id()<< " EEDetId = " << eeid.ix() << " " << eeid.iy() << "\n"	
      << " Time = " << isim->time() << "\n"
      << " Track Id = " << isim->geantTrackId() << "\n"
      << " Energy = " << isim->energy();
    
    uint32_t crystid = eeid.rawId();
     
    if (eeid.zside() > 0 ) {
      nEEzpHits++;
      EEetzp_ += isim->energy();
      eemapzp[crystid] += isim->energy();
      if (meEEzpHitEnergy_) meEEzpHitEnergy_->Fill(isim->energy());
      if (meEEzpOccupancy_) meEEzpOccupancy_->Fill(eeid.ix(), eeid.iy());
    }
    else if (eeid.zside() < 0 ) {
      nEEzmHits++;
      EEetzm_ += isim->energy();
      eemapzm[crystid] += isim->energy();
      if (meEEzmHitEnergy_) meEEzmHitEnergy_->Fill(isim->energy());
      if (meEEzmOccupancy_) meEEzmOccupancy_->Fill(eeid.ix(), eeid.iy());
    }
    
    eemap[crystid] += isim->energy();
  }
  
  if (meEEzpCrystals_) meEEzpCrystals_->Fill(eemapzp.size());
  if (meEEzmCrystals_) meEEzmCrystals_->Fill(eemapzm.size());
    
  if (meEEzpHits_)    meEEzpHits_->Fill(nEEzpHits);
  if (meEEzmHits_)    meEEzmHits_->Fill(nEEzmHits);
    
  
  int nEEHits = nEEzmHits + nEEzpHits;
  if (nEEHits > 0) {
    
    uint32_t  eecenterid = getUnitWithMaxEnergy(eemap);
    EEDetId myEEid(eecenterid);
    int bx = myEEid.ix();
    int by = myEEid.iy();
    int bz = myEEid.zside();
    ee1 =  energyInMatrixEE(1,1,bx,by,bz,eemap);
    if (meEEe1_) meEEe1_->Fill(ee1);
    ee9 =  energyInMatrixEE(3,3,bx,by,bz,eemap);
    if (meEEe9_) meEEe9_->Fill(ee9);
    ee25=  energyInMatrixEE(5,5,bx,by,bz,eemap);
    if (meEEe25_) meEEe25_->Fill(ee25);
    
    MapType  neweemap;
    if( fillEEMatrix(3,3,bx,by,bz,neweemap, eemap)){
      ee4 = eCluster2x2(neweemap);
      if (meEEe4_) meEEe4_->Fill(ee4);
    }
    if( fillEEMatrix(5,5,bx,by,bz,neweemap, eemap)){
      ee16 = eCluster4x4(ee9,neweemap); 
      if (meEEe16_) meEEe16_->Fill(ee16);
    }
    
    if (meEEe1oe4_   && ee4  > 0.1 ) meEEe1oe4_  ->Fill(ee1/ee4);
    if (meEEe4oe9_   && ee9  > 0.1 ) meEEe4oe9_  ->Fill(ee4/ee9);
    if (meEEe9oe16_  && ee16 > 0.1 ) meEEe9oe16_ ->Fill(ee9/ee16);
    if (meEEe16oe25_ && ee25 > 0.1 ) meEEe16oe25_->Fill(ee16/ee25);
    if (meEEe1oe25_  && ee25 > 0.1 ) meEEe1oe25_ ->Fill(ee1/ee25);
    if (meEEe9oe25_  && ee25 > 0.1 ) meEEe9oe25_ ->Fill(ee9/ee25);
    
  }
  
  
  if ( MyPEcalValidInfo->ee1x1() > 0. ) {
    std::vector<float>  EX0 = MyPEcalValidInfo->eX0();
    for (int ii=0;ii< 26;ii++ ) {
      if (meEELongitudinalShower_) meEELongitudinalShower_->Fill(float(ii), EX0[ii]);
    }
  }
  
}

float EcalEndcapSimHitsValidation::energyInMatrixEE(int nCellInX, int nCellInY,
                                        int centralX, int centralY,
                                        int centralZ, MapType& themap){
  
  int   ncristals   = 0;
  float totalEnergy = 0.;
        
  int goBackInX = nCellInX/2;
  int goBackInY = nCellInY/2;
  int startX    = centralX-goBackInX;
  int startY    = centralY-goBackInY;
  
  for (int ix=startX; ix<startX+nCellInX; ix++) {
    for (int iy=startY; iy<startY+nCellInY; iy++) {
      
      uint32_t index ;
      try {
        index = EEDetId(ix,iy,centralZ).rawId();
      } catch ( cms::Exception &e ) { continue ; }
      totalEnergy   += themap[index];
      ncristals     += 1;
    }
  }
  
  LogDebug("GeomInfo")
    << nCellInX << " x " << nCellInY 
    << " EE matrix energy = " << totalEnergy
    << " for " << ncristals << " crystals";
  return totalEnergy;
  
}

bool  EcalEndcapSimHitsValidation::fillEEMatrix(int nCellInX, int nCellInY,
                                    int CentralX, int CentralY,int CentralZ,
                                    MapType& fillmap, MapType&  themap)
{
   int goBackInX = nCellInX/2;
   int goBackInY = nCellInY/2;

   int startX  =  CentralX - goBackInX;
   int startY  =  CentralY - goBackInY;

   int i = 0 ;
   for ( int ix = startX; ix < startX+nCellInX; ix ++ ) {

      for( int iy = startY; iy < startY + nCellInY; iy++ ) {

        uint32_t index ;
        try {
          index = EEDetId(ix,iy,CentralZ).rawId();
        } catch ( cms::Exception &e ) { continue ; }
        fillmap[i++] = themap[index];
      }
   }
   uint32_t  centerid = getUnitWithMaxEnergy(themap);

   if ( fillmap[i/2] == themap[centerid] ) 
        return true;
   else
        return false;
}


float EcalEndcapSimHitsValidation::eCluster2x2( MapType& themap){
  float  E22=0.;
  float e012  = themap[0]+themap[1]+themap[2];
  float e036  = themap[0]+themap[3]+themap[6];
  float e678  = themap[6]+themap[7]+themap[8];
  float e258  = themap[2]+themap[5]+themap[8];

  if ( (e012>e678 || e012==e678) && (e036>e258  || e036==e258))
    return     E22=themap[0]+themap[1]+themap[3]+themap[4];
  else if ( (e012>e678 || e012==e678)  && (e036<e258 || e036==e258) )
    return    E22=themap[1]+themap[2]+themap[4]+themap[5];
  else if ( (e012<e678 || e012==e678) && (e036>e258 || e036==e258))
    return     E22=themap[3]+themap[4]+themap[6]+themap[7];
  else if ( (e012<e678|| e012==e678)  && (e036<e258|| e036==e258) )
    return    E22=themap[4]+themap[5]+themap[7]+themap[8];
  else {
    return E22;
  }
}

float EcalEndcapSimHitsValidation::eCluster4x4(float e33,  MapType&  themap){
  float E44=0.;
  float e0_4   = themap[0]+themap[1]+themap[2]+themap[3]+themap[4];
  float e0_20  = themap[0]+themap[5]+themap[10]+themap[15]+themap[20];
  float e4_24  = themap[4]+themap[9]+themap[14]+themap[19]+themap[24];
  float e0_24  = themap[20]+themap[21]+themap[22]+themap[23]+themap[24];
  
  if ((e0_4>e0_24 || e0_4==e0_24) && (e0_20>e4_24|| e0_20==e4_24))
    return E44=e33+themap[0]+themap[1]+themap[2]+themap[3]+themap[5]+themap[10]+themap[15];
  else if ((e0_4>e0_24 || e0_4==e0_24)  && (e0_20<e4_24 || e0_20==e4_24))
    return E44=e33+themap[1]+themap[2]+themap[3]+themap[4]+themap[9]+themap[14]+themap[19];
  else if ((e0_4<e0_24|| e0_4==e0_24) && (e0_20>e4_24 || e0_20==e4_24))
    return E44=e33+themap[5]+themap[10]+themap[15]+themap[20]+themap[21]+themap[22]+themap[23];
  else if ((e0_4<e0_24|| e0_4==e0_24) && (e0_20<e4_24 || e0_20==e4_24))
    return E44=e33+themap[21]+themap[22]+themap[23]+themap[24]+themap[9]+themap[14]+themap[19];
  else{
    return E44;
  }
}

uint32_t EcalEndcapSimHitsValidation::getUnitWithMaxEnergy(MapType& themap) {
  
  //look for max
  uint32_t unitWithMaxEnergy = 0;
  float    maxEnergy = 0.;
  
  MapType::iterator iter;
  for (iter = themap.begin(); iter != themap.end(); iter++) {
    
    if (maxEnergy < (*iter).second) {
      maxEnergy = (*iter).second;       
      unitWithMaxEnergy = (*iter).first;
    }                           
  }
  
  LogDebug("GeomInfo")
    << " max energy of " << maxEnergy 
    << " GeV in Unit id " << unitWithMaxEnergy;
  return unitWithMaxEnergy;
}


