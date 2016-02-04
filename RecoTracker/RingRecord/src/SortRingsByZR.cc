//
// Package:         RecoTracker/RingRecord
// Class:           SortRingsByZR
// 
// Description:     sort rings by ZR
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Dec 20 17:31:01 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/02/05 19:10:04 $
// $Revision: 1.1 $
//


#include "RecoTracker/RingRecord/interface/SortRingsByZR.h"

SortRingsByZR::SortRingsByZR() {
  //
  // default constructor
  //
}

SortRingsByZR::~SortRingsByZR() { 
  //
  // default destructor
  //
}

bool SortRingsByZR::RingsSortedInZR(const Ring *a, const Ring *b) const {
  //
  // return true if rings are sorted in abs ZR, else return false
  //

  // return value
  bool result = true;

  if ( a->getindex() > b->getindex() ) {
    result = false;
  }

//   double r1 = (a->getrmax() - a->getrmin())/2 + a->getrmin();
//   double z1 = (a->getzmax() - a->getzmin())/2 + a->getzmin();

//   double r2 = (b->getrmax() - b->getrmin())/2 + b->getrmin();
//   double z2 = (b->getzmax() - b->getzmin())/2 + b->getzmin();

//   if ( a->getType() == b->getType() ) {
//     if ( a->getType() == Ring::TIBRing ||
// 	 a->getType() == Ring::TOBRing ||
// 	 a->getType() == Ring::PXBRing ) {
//       if ( r1 > r2 ) {
// 	result = false;
//       } else {
// 	if ( z1 > z2 ) {
// 	  result = false;
// 	}
//       }
//     } else {
//       if ( z1 > z2 ) {
// 	result = false;
//       } else {
// 	if ( r1 > r2 ) {
// 	  result = false;
// 	}
//       }
//     }
//   } else {
//     if ( (a->getType() == Ring::PXBRing && b->getType() == Ring::TIBRing) ||
// 	 (a->getType() == Ring::TIBRing && b->getType() == Ring::PXBRing) ) {
//       if ( r1 > r2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( z1 > z2 ) {
// // 	  result = false;
// // 	}
// //       }
      
//     } else if ( (a->getType() == Ring::PXBRing && b->getType() == Ring::TOBRing) ||
// 		(a->getType() == Ring::TOBRing && b->getType() == Ring::PXBRing) ) {
//       if ( r1 > r2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( z1 > z2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::PXBRing && b->getType() == Ring::PXFRing) ||
// 		(a->getType() == Ring::PXFRing && b->getType() == Ring::PXBRing) ) {
//       if ( z1 > z2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( r1 > r2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::PXBRing && b->getType() == Ring::TIDRing) ||
// 		(a->getType() == Ring::TIDRing && b->getType() == Ring::PXBRing) ) {
//       if ( z1 > z2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( r1 > r2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::PXBRing && b->getType() == Ring::TECRing) ||
// 		(a->getType() == Ring::TECRing && b->getType() == Ring::PXBRing) ) {
//       if ( z1 > z2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( r1 > r2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::TIBRing && b->getType() == Ring::TOBRing) ||
// 		(a->getType() == Ring::TOBRing && b->getType() == Ring::TIBRing) ) {
//       if ( r1 > r2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( z1 > z2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::TIBRing && b->getType() == Ring::PXFRing) ||
// 		(a->getType() == Ring::PXFRing && b->getType() == Ring::TIBRing) ) {
//       if ( r1 > r2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( z1 > z2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::TIBRing && b->getType() == Ring::TIDRing) ||
// 		(a->getType() == Ring::TIDRing && b->getType() == Ring::TIBRing) ) {
//       if ( z1 > z2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( r1 > r2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::TIBRing && b->getType() == Ring::TECRing) ||
// 		(a->getType() == Ring::TECRing && b->getType() == Ring::TIBRing) ) {
//       if ( z1 > z2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( r1 > r2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::TOBRing && b->getType() == Ring::PXFRing) ||
// 		(a->getType() == Ring::PXFRing && b->getType() == Ring::TOBRing) ) {
//       if ( r1 > r2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( z1 > z2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::TOBRing && b->getType() == Ring::TIDRing) ||
// 		(a->getType() == Ring::TIDRing && b->getType() == Ring::TOBRing) ) {
//       if ( r1 > r2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( z1 > z2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::TOBRing && b->getType() == Ring::TECRing) ||
// 		(a->getType() == Ring::TECRing && b->getType() == Ring::TOBRing) ) {
//       if ( z1 > z2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( r1 > r2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::PXFRing && b->getType() == Ring::TIDRing) ||
// 		(a->getType() == Ring::TIDRing && b->getType() == Ring::PXFRing) ) {
//       if ( z1 > z2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( r1 > r2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::PXFRing && b->getType() == Ring::TECRing) ||
// 		(a->getType() == Ring::TECRing && b->getType() == Ring::PXFRing) ) {
//       if ( z1 > z2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( r1 > r2 ) {
// // 	  result = false;
// // 	}
// //       }

//     } else if ( (a->getType() == Ring::TIDRing && b->getType() == Ring::TECRing) ||
// 		(a->getType() == Ring::TECRing && b->getType() == Ring::TIDRing) ) {
//       if ( z1 > z2 ) {
// 	result = false;
//       }
// //  else {
// // 	if ( r1 > r2 ) {
// // 	  result = false;
// // 	}
// //       }

//     }

//   }

//   if ( a->getindex() == 15 || b->getindex() == 15 ) {
//     std::cout << "a: " << a->getindex() << " b: " << b->getindex() << std::endl;
//     std::cout << "r1: " << r1 << " r2: " << r2 << " z1: " << z1 << " z2: " << z2 << std::endl;
//     std::cout << "result: " << result << std::endl;
//   }

  return result;
}
