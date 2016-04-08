/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*
****************************************************************************/

#ifndef DataFormats_TotemRPReco_RPTimingDetectorHit
#define DataFormats_TotemRPReco_RPTimingDetectorHit

#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class RPTimingDetectorHit {
public:
    RPTimingDetectorHit(const unsigned int &det_id = 0, unsigned short electrode_id = 0,
                        const Local2DPoint position = Local2DPoint(0, 0)) : det_id(det_id),
                                                        electrode_id(electrode_id),
                                                        position(position) { }

    unsigned int GetDetId() const { return det_id; }
    unsigned short GetElectrodeId() const { return electrode_id; }
    const Local2DPoint &GetPosition() const {
        return position;
    }

private:
    unsigned int det_id;
    unsigned short electrode_id;
    Local2DPoint position;
};


inline bool operator<(const RPTimingDetectorHit & a, const RPTimingDetectorHit & b) {
    if (a.GetDetId() == b.GetDetId()) {
        return a.GetElectrodeId() < b.GetElectrodeId();
    }

    return a.GetDetId() < b.GetDetId();
}

#endif
