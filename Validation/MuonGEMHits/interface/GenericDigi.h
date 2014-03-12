#ifndef GEMValidation_GenericDigi_h
#define GEMValidation_GenericDigi_h

/**\file GenericDigi

 Description: generic digi structure with its accessors and containers.

 Can use it with:
 - GEM strip, pad and co-pad digis
 - CSC comparator(strip) and wire digis
 - CSC trigger stub digis

 Original Author:  "Vadim Khotilovich"
 $Id:$
*/

#include "Validation/MuonGEMHits/interface/BaseMatcher.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <vector>
#include <tuple>
#include <iostream>

namespace matching {

typedef enum {INVALID=0, GEM_STRIP, GEM_PAD, GEM_COPAD, CSC_STRIP, CSC_WIRE, CSC_CLCT, CSC_ALCT, CSC_LCT} DigiType;

// digi info keeper: <detid, channel, bx, type, quality, bend, WireGroup>
typedef std::tuple<unsigned int, int, int, DigiType, int, int, int, float> Digi;

// digi collection
typedef std::vector<Digi> DigiContainer;

// digi makeres
inline Digi make_digi() { return std::make_tuple(0, 0, 0, INVALID, 0, 0, 0, 0.); }
inline Digi make_digi(unsigned int id, int ch, int bx, DigiType t) { return std::make_tuple(id, ch, bx, t, 0, 0, 0, 0.); }
inline Digi make_digi(unsigned int id, int ch, int bx, DigiType t, int q) { return std::make_tuple(id, ch, bx, t, q, 0, 0, 0.); }
inline Digi make_digi(unsigned int id, int ch, int bx, DigiType t, int q, int pat) { return std::make_tuple(id, ch, bx, t, q, pat, 0, 0.); }
inline Digi make_digi(unsigned int id, int ch, int bx, DigiType t, int q, int pat, int wg) { return std::make_tuple(id, ch, bx, t, q, pat, wg, 0.); }
inline Digi make_digi(unsigned int id, int ch, int bx, DigiType t, int q, int pat, int wg, float dphi) { return std::make_tuple(id, ch, bx, t, q, pat, wg, dphi); }

// digi accessors
inline bool is_valid(const Digi& d) {return std::get<0>(d) != INVALID; }

inline unsigned int digi_id(const Digi& d) { return std::get<0>(d); }
inline int digi_channel(const Digi& d) { return std::get<1>(d); }
inline int digi_bx(const Digi& d) { return std::get<2>(d); }
inline DigiType digi_type(const Digi& d) { return std::get<3>(d); }
inline int digi_quality(const Digi& d) { return std::get<4>(d); }
inline int digi_pattern(const Digi& d) { return std::get<5>(d); }

// can access and also modify the WG value by reference with this one
int& digi_wg(Digi& d);
// only read for const digi
int digi_wg(const Digi& d);

// can access and also modify the dphi value by reference with this one
inline float& digi_dphi(Digi& d) { return std::get<7>(d); }
// only read for const digi
inline float digi_dphi(const Digi& d) { return std::get<7>(d); }

}

std::ostream & operator<<(std::ostream & o, const matching::Digi& d);

#endif

