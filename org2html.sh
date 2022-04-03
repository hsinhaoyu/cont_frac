#!/bin/bash

echo $@

/Applications/Emacs.app/Contents/MacOS/Emacs -Q --batch --eval "
    (progn
      (dolist (file command-line-args-left)
        (with-current-buffer (find-file-noselect file)
          (org-html-export-to-html))))
  " "$@"
