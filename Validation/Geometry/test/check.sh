#!/bin/bash
{
python3 -c "from FWCore.PythonFramework.CmsRun import CmsRun" 2>/dev/null &&echo "using python3"
}||
{
python -c "from FWCore.PythonFramework.CmsRun import CmsRun" 2>/dev/null && echo "using python2"
}
