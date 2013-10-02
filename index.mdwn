---
title: CMS Offline Software
layout: default
related:
 - { name: Project page, link: "https://github.com/cms-sw/cmssw" }
 - { name: Feedback, link: "https://github.com/cms-sw/cmssw/issues/new" }
---

# Welcome to CMSSW on GitHub pages

CMS is migrating the repository of its Offline Experiment Software (CMSSW) to
git and github.

# Migration schedule:

### CMSSW_7_0_X

On **June 26th**, in line with what is `CMSSW_6_2_0_pre8`, `CMSSW_7_0_X`
branch will open in GitHub.

The current, test, [cms-sw/cmssw](http://github.com/cms-sw/cmssw) repository
will be scrapped and started from scratch! **You'll need to do the same for your
forks!**

### CMSSW_6_2_X, CMSSW_5_3_X, CMSSW_4_4_X, CMSSW_4_1_X, SLHC releases.

On the day `CMSSW_6_2_0` was released, we opened branches in git for all the
required stable release queue.

CVS access to CMSSW repository became read only at that point and all the
development moved to git.

### UserCode

You will be able to commit to the UserCode repository until the 15th of
October, in order to avoid disruption for the summer conferences.

This is the final date for your usercode migration, no further grace periods
will be provided by CERN IT.

Instructions on how to migrate CMSSW releases can be found
[here](http://cms-sw.github.io/cmssw/usercode-faq.html).

[cvs2git]: http://cvs2svn.tigris.org/cvs2git.html
