---
title: CMS Offline Software
layout: default
related:
 - { name: Home, link: index.html }
 - { name: Project, link: https://github.com/cms-sw/cmssw }
 - { name: Topic Collector, link: https://cern.ch/cmsgit/cmsgit}
 - { name: Feedback, link: https://github.com/cms-sw/cmssw/issues/new }
---

# Troubleshooting git

### Sparse checkout does not work.

Apparently some university deployed a non working `git` 1.7.4 client. This
results in sparse checkout misbehavior. Using 1.7.4.1 or later seems to fix the
issue.

### Missing public key.

In case you get the following message:

    Permission denied (publickey).
    fatal: The remote end hung up unexpectedly

you forgot to register your ssh key when you registered to github, you can do
it by going to <https://github.com/settings/ssh> and adding your public key
there.


