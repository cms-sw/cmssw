#!/bin/csh -f

eval `scramv1 runtime -csh`

##################################################
echo "===========> Validating reconstruction....."
cmsRun val-reco.cfg

echo "===========> compare Track plots..."
root -b -p -q DoCompare.C\(\"pixelTrackHistos_Val\",\"../data/pixelTrackHistos\"\)

echo "===========> compare Vertex plots..."
root -b -p -q DoCompare.C\(\"pixelVertexHistos_Val\",\"../data/pixelVertexHistos\"\)


##################################################
echo "===========> Validating I/O....."
cmsRun val-io.cfg

echo "===========> compare Track plots..."
root -b -p -q DoCompare.C\(\"pixelTrackHistos_ioVal\",\"pixelTrackHistos_Val\"\)

echo "===========> compare Vertex plots..."
root -b -p -q DoCompare.C\(\"pixelVertexHistos_ioVal\",\"pixelVertexHistos_Val\"\)
