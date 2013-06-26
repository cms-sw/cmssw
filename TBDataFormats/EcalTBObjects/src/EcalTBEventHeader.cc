#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"
#include <cstdio>

//! Return the event type: "beam", "laser", "pedestal". "error"
//! or a number orresponding to the orginal eventype stored in 
//! the RRF. 
std::string EcalTBEventHeader::eventType() const
{
  // this piece of code was in TPilot (ievtype)
  int m = triggerMask_ & 0x00FFFF01; // bits 0, 8..23

  // this piece of code was in TPilot (numbit)
  int w = m ;
  int num = 0;
  if ( w < 0 ) {        //
    w &= 0x7FFFFFFF;    // CINT Error
    num++;              //
  }
  do { num += ( w & 0x1 ); } while ( (w >>= 1) != 0 );
  // end of numbit

  if (num != 1)  return std::string("error") ;

  char chEvtype[80] ;
  if ( m == 1 ) return std::string("beam") ; // Physics triggers
  for(int i=0;i<24;i++) {
    if ( ( m & 0x1 ) == 1 ) {
      sprintf(chEvtype, "%d", i) ;
      if (i==11) return std::string("pedestal") ;
      if (i==13) return std::string("laser") ;
      return std::string(chEvtype) ;
    }
    m >>= 1;
  }
  return std::string("error");

  //   // FIXME: to be uncommented with new rawroot
  //   int evtype = rawHeader->GetEventType() ;
  //   if (evtype == 0) return std::string("error") ;
  //   if (evtype == 1) return std::string("beam") ; 
  //   if (evtype == 11) return std::string("pedestal") ;
  //   if (evtype == 13) return std::string("laser") ;
  //   char chEvtype[80] ;
  //   sprintf(chEvtype, "%d", evtype) ;
  //   return std::string(chEvtype) ;
}


int EcalTBEventHeader::dbEventType() const{
  std::string evtType = eventType();
  int ievtType = 0;
  if (evtType=="beam") ievtType = 1 ;
  if (evtType=="laser") ievtType = 2 ;
  if (evtType=="pedestal") ievtType = 1 ; // same as beam
  if (ievtType == 2) {
    LaserType laser_type = laserTypeName();
    //if (laser_type == EcalTBEventHeader::LBlue) ievtType += 0 ;
    if (laser_type == EcalTBEventHeader::LGreen) ievtType += 1 ;
    if (laser_type == EcalTBEventHeader::LInfrared) ievtType += 2 ;
    if (laser_type == EcalTBEventHeader::LRed) ievtType += 3 ;
  }
  return ievtType;
}

std::ostream& operator<<(std::ostream& s, const EcalTBEventHeader& eventHeader) {
  s << "Run Number " << eventHeader.runNumber() << " Event Number " << eventHeader.eventNumber() << " Burst Number " << eventHeader.burstNumber() ;
  return s;
}
