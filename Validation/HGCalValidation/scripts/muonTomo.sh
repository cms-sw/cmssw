#!/bin/bash
geom=D110
cmsRun testHGCalSingleMuonPt100_cfg.py geometry=$geom
cmsRun testHGCalCellHitSum_cfg.py geometry=$geom
