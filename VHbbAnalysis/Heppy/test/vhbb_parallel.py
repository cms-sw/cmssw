#! /usr/bin/env python
from vhbb_combined import *
from PhysicsTools.Heppy.utils.cmsswPreprocessor import CmsswPreprocessor
import glob, sys

if __name__ == "__main__":
    fi = sys.argv[1]
    outfn = sys.argv[2]

    selectedComponents = [
        cfg.MCComponent(
            files = [fi],
            name = fi.split("/")[-1].split(".")[0]
        )
    ]
    config.components = selectedComponents
    from PhysicsTools.HeppyCore.framework.looper import Looper 
    looper = Looper('Loop_{0}'.format(outfn), config, nPrint=0)
    looper.loop() 
    looper.write()
