---
title: CMS Offline Software
layout: default
related:
 - { name: Project page, link: 'https://github.com/cms-sw/cmssw' }
 - { name: Feedback, link: 'https://github.com/cms-sw/cmssw/issues/new' }
---
# Interaction with old CVS

As you might have noticed, not all the per package tags have been imported into
the main CMSSW git repository. In particular per-package tags and branches are
not going to be migrated automatically, to avoid overpopulating the repository.

On the other hand we do want to simplify the job of porting those tags from CVS
to git we have prepared a few tools which will hopefully make the task easy.
 
First of all, a read only, immutable, version of every package in CMSSW
packages  will be available as a standalone git repository at:

    https://github.com/cms-cvs-history/<subsystem>-<package>

Where `<subsystem>` and `<package>` are as found in CVS. E.g.:

    https://github.com/cms-cvs-history/DQMServices-Core

These repository will contain every tag and branch for the given package. Just
like you used to find them when navigating in CVS. The only difference being
that each tag is now prepended with `<subsystem>-<package>-` to simplify
importing them in the main git repository.

In order to simplify import from CVS, we also provide a wrapper script: `git
cms-cvs-history`, which can be used to navigate and possibly import per package
tags into your favorite CMSSW git branch.

### Listing the tags available in CVS for a given package.

    git cms-cvs-history tags <package>

where `<package>` is in the usual form `Subsystem/Package`, e.g.:

    git cms-cvs-history tags Configuration/Generator

### Importing a given tag for a given package

    git cms-cvs-history import <tag> <package>

E.g.:

    git cms-cvs-history import V01-15-12 Configuration/Generator

### Diffing two tags:

    git cms-cvs-history diff <cvs-tag1>..<cvs-tag2> <package>

equivalent to `cvs diff -r <cvs-tag1> -r <cvs-tag2> <package>`.

### Package log:

    git cms-cvs-history log <branch> <package>

Whole history of a given `<branch>`.

* auto-gen TOC:
{:toc}
