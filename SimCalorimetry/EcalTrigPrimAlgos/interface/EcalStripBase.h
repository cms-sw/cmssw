#ifndef ECAL_STRIP_BASE_H
#define ECAL_STRIP_BASE_H

#include <Calorimetry/CaloDetector/interface/CaloBase.h>
#include <Calorimetry/CaloDetector/interface/CellProperties.h>


// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;

/*
 * To be created
 * should allow to determine
 * - which crystal belongs to which strip
 * - which strip to which tower
 */
class EcalStripBase : public CaloBase {


 public:
  CellProperties myCellProperties;
};


#endif
