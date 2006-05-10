#ifndef ECAL_FENIX_TCP_FORMAT_H
#define ECAL_FENIX_TCP_FORMAT_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFormatter.h>
#include <boost/cstdint.hpp>


namespace tpg {

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
   \class EcalFenixStripFormat
   \brief Formatting for Fenix strip
   *  input 10 bits from Ettot 
   *         1 bit from fgvb
   *         3 bits TriggerTowerFlag (dummy for the moment)
   *  output: 16 bits
   *  simple formatting
   *  
   */
class EcalFenixTcpFormat : public EcalVFormatter {


 public:
    EcalFenixTcpFormat();
    virtual ~EcalFenixTcpFormat();
    virtual vector<int> process(vector<int>,vector<int>);
   };

} /* End of namespace tpg */

#endif
