#!/bin/zsh
# Opens the presentation and the local SentinelAI app for the recorded demo.
set -eu
cd -- "${0:A:h}"

slides_url="$(/usr/bin/python3 -c 'from pathlib import Path; print(Path("delivery/Sunumu_Ac.html").resolve().as_uri() + "#slayt-4")')"
open_demo() {
  open -a Safari "$slides_url"
  open -a Safari "$app_url"
}

for port in {8501..8510}; do
  app_url="http://127.0.0.1:$port"
  pid="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1)"
  if [[ -z "$pid" ]]; then
    break
  fi
  if lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | grep -Fxq "n$PWD" &&
     [[ "$(curl --noproxy '*' -fsS --max-time 2 "$app_url/_stcore/health" 2>/dev/null)" == ok ]]; then
    open_demo
    exit 0
  fi
done
if [[ -n "$pid" ]]; then
  print -u2 "8501–8510 arasında boş port bulunamadı."
  exit 1
fi

print "SentinelAI açılıyor. Demo boyunca bu Terminal penceresini açık bırak."
STREAMLIT_SERVER_HEADLESS=true STREAMLIT_SERVER_PORT="$port" zsh Baslat.command &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM HUP
until [[ "$(curl --noproxy '*' -fsS --max-time 2 "$app_url/_stcore/health" 2>/dev/null)" == ok ]]; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    exit 1
  fi
  sleep 1
done
open_demo
wait "$server_pid"
