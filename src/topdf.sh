#!/usr/bin/env bash
# Print an HTML file to PDF with headless Chrome: topdf.sh in.html out.pdf
set -euo pipefail
in=$(realpath "$1"); out=$(realpath -m "$2")
google-chrome --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer \
  --virtual-time-budget=20000 --print-to-pdf="$out" "file://$in" 2>/dev/null
ls -la "$out"
