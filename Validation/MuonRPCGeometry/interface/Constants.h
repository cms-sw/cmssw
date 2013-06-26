#ifndef RPCPatts_Constants_h
#define RPCPatts_Constants_h
// -*- C++ -*-
//
// Package:     RPCPatts
// Class  :     constants
// 
/**\class constants constants.h L1Trigger/RPCPatts/interface/constants.h

 Description: Contains some constants ;)

 Usage:
    <usage>

*/
//
// Original Author:  TMF
//         Created:  Wed Oct  3 10:28:12 CEST 2007
// $Id: Constants.h,v 1.1 2009/02/05 10:09:55 fruboes Exp $
//
#include <cmath>
namespace RPCpg {
  const int mu = 0;
  const int mubar = 1;
  const int muundefined = 10;
  
  const unsigned int maxPlanes_s = 6;
  const unsigned int empty_s = 99; // note NOT_CONNECTED?
  const unsigned int ptBins_s = 32;
  
  static const double pts[33] = {
     0.0,  0.01,
     1.5,  2.0, 2.5,  3.0,  3.5,  4.0,  4.5,
     5.,   6.,   7.,   8.,
     10.,  12., 14.,  16.,  18.,
     20.,  25.,  30., 35.,  40.,  45.,
     50.,  60.,  70., 80.,  90.,  100., 120., 140., 160.};
   double rate(double x); // rate from pt = x [Gev/c] to inf
  
}

#include <vector>
#include <map>
#include <string>
namespace RPCPatGen{

      // Quality
   struct TQualityStruct{
          TQualityStruct(const std::string &str, short int qual, short int tabNum):
                m_qualStr(str),m_qual(qual),m_tabNum(tabNum){};
          std::string m_qualStr;
          short int m_qual;
          short int m_tabNum;
   };
   typedef std::vector<TQualityStruct> TQualVec;
   typedef std::map<int,TQualVec> TTowerToQualMap;
      
   typedef std::vector<int> TOrSize;
   typedef std::map<int, TOrSize> TOrSizeMap; /// key - tower number, value - vector of ORSizes
   typedef std::map<int, TOrSizeMap> TPtOrSizeMap;/// key - ptCode, value - TOrSize

}


#endif
