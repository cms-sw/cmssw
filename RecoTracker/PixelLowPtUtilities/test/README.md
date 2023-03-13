### How to produce a calibration file for ClusterShapeHitFilter

1. Copy the cfg file in your area

    ```shell
    cp $CMSSW_RELEASE_BASE/src/RecoTracker/PixelLowPtUtilities/test/clusterShapeExtractor_phase* .
    ```
2. Get the list of files from relval and substitute to those in the file in the release
3. cmsRun it: it create a root file named _clusterShape.root_
4. Produce a calibration file:

    ```shell
    clusterShapeAnalyzer --input clusterShape.root --output myClusterShape.par
    ```
5.  you can produce also a calibration file w/o taking BPIX1 into account : just change
    ```code
    process.clusterShapeExtractor.noBPIX1=False
    ```
    to
    ```code
    process.clusterShapeExtractor.noBPIX1=True
    ```

6. Analysis the resulting file using/modifing the pcsfVerify.ipynb notebook
7. to use the produced file(s) edit RecoTracker/PixelLowPtUtilities/python/ClusterShapeHitFilterESProducer_cfi.py
