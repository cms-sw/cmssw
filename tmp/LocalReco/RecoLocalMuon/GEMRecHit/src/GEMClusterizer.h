#ifndef RecoLocalMuon_GEMClusterizer_h
#define RecoLocalMuon_GEMClusterizer_h
/** \class GEMClusterizer
 *  $Date: 2006/07/16 07:25:39 $
 *  $Revision: 1.5 $
 *  \author M. Maggi -- INFN Bari
 */

#include "GEMClusterContainer.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

class GEMCluster;
class GEMClusterizer{
 public:
  GEMClusterizer();
  ~GEMClusterizer();
  GEMClusterContainer doAction(const GEMDigiCollection::Range& digiRange);

 private:
  GEMClusterContainer doActualAction(GEMClusterContainer& initialclusters);

 private:
  GEMClusterContainer cls;
};
#endif
