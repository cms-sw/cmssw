How to make a regression test of propagation

1) run cmsRun rkTest_cfg.py >& baseline.log&
2) run it in the new version as cmsRun rkTest_cfg.py >& newversion.log&
3) take a diff with python findDiff.py baseline.log newversion.log >& diff.txt
4) look at the diff
