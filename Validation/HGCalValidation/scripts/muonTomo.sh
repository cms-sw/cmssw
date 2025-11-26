#!/bin/bash
geom=D121
cmsRun testHGCalSingleMuonPt100_cfg.py geometry=$geom
cmsRun testHGCalCellHitSum_cfg.py geometry=$geom
