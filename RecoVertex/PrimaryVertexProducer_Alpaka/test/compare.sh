#!/bin/sh

cmsRun ../../PrimaryVertexProducer/test/testPrimaryVertexProducer_CPU.py

cmsRun testPrimaryVertexProducer_Alpaka.py --backend $i

python compareAlgos.py testAlpaka.root testCPU.root
