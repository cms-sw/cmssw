/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/01/12 12:16:26 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "RPCRecHitQuality.h"



//#include "DTHitQualityUtils.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"



#include "Histograms.h"



#include "TFile.h"

#include <iostream>
#include <map>



using namespace std;
using namespace edm;



