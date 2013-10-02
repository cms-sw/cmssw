---
title: CMS Offline Software
layout: default
related:
 - { name: Project page, link: 'https://github.com/cms-sw/cmssw' }
 - { name: Feedback, link: 'https://github.com/cms-sw/cmssw/issues/new' }
---

# CMS - Software development tools projects

- [PKGTOOLS](https://github.com/cms-sw/pkgtools): CMS packaging and distribution utility
- [CMSDIST](https://github.com/cms-sw/cmsdist): distribution recipes
- [SCRAM](https://github.com/cms-sw/SCRAM): CMS build tool
- [CMSSW config](https://github.com/cms-sw/cmssw-config): CMSSW build configuration

# Migration information about CMSSDT projects

## PKGTOOLS (DONE)

Four branches created `V00-16-XX`, `V00-20-XX`, `V00-21-XX`, `V00-22-XX`,
corresponding to the latest greatest tag in their own series.

## CMSDIST (DONE)

All tags migrated. Files in CVS Attic **NOT** are not migrated. A few old large
files are not migrated. If you need a tag which has files in the Attic, you'll
have to branch such a tag, import the missing files and move the tag.

From now on the `master` branch will be used for the production architecture of
the latest development series. Work on experimental architectures or older
release series will go into branches named
`CMSSW_<major>_<minor>_X/<architeure>`.  Experimental architectures in
particular should make an effor to merge their changes back into the master
branch.

Changes must be provided in form of pull requests to the appropriate branch.

CVS HEAD when the import happened is renamed to import-HEAD.

## SCRAM (DONE)

All relevant tags migrated. Next release will have checkout from the git
release area.

## CMSSW config (DONE)

All tags migrated. `scram-project-build.file` revision 1.134 has the required
changes.
