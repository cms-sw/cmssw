---
title: CMS Offline Software
layout: default
related:
 - { name: Project page, link: 'https://github.com/cms-sw/cmssw' }
 - { name: Feedback, link: 'https://github.com/cms-sw/cmssw/issues/new' }
---

## Tutorial: analysis recipe

This tutorial will guide you through the creation of a topic branch for your
analysis, and how to share it with your coworkers.

### Disclaimer

The instructions contained in this tutorial will allow you to create a **fake**
analysis. For your real work, please use the branches suggested by your
analysis coordinator.

### Before you start.

Please make sure you registered to GitHub and that you have provided them
a ssh public key to access your private repository. For more information see
the [FAQ](faq.html).

### The analysis use case.

The analysis workflow usually depends on changes not released in an official
CMSSW builds. The old recipes where of the kind:

1. Get a given release.
2. Checkout a few, non released, packages as suggested by the Analysis convenors.
3. (Optionally) checkout a few packages from your user area.

The equivalent which are going to use for git is:

1. Get a given release.
2. Merge in your area a few, non released, topic branches, as suggested by the
   analysis convenors.
3. (Optionally) merge your private code:
    * From a your private fork of CMSSW.
    * From from your usercode directory.

### Get your release

This is done with the usual:

    scram project CMSSW_7_0_0_pre0
    cd CMSSW_7_0_0_pre0/src
    eval `scram run -sh`


