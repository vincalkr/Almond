#!/bin/sh
# Stops every Almond process of this folder that ./almond serve started: the servers (DeepSeek, GLM, Qwen, Qwen27,
# Hy4 and the router), their launch.py wrappers and the engine / image-encoder workers, the way Ctrl+C would.
# ./almond serve switches power-tune off once no server is left; --force then kills what is left.
# Waits up to TIMEOUT seconds (default 60).
#   ./stop.sh            stop and wait
#   ./stop.sh --force    the same, then kill a server or worker still running after TIMEOUT
R=$(cd "$(dirname "$0")" && pwd) || exit 1
FORCE=0; [ "${1:-}" = --force ] && FORCE=1
TIMEOUT=${TIMEOUT:-60}

# this folder's servers and router only (another checkout's are left alone); keep in sync with launch.py's SERVERS
servers() { pgrep -f "$R/tools/(v41/serve/serve_ds41|glm/serve/serve_glm|qwen/serve/serve_qwen|hy4/serve/serve_hy4|inkling/serve/serve_inkling|router/almond_router)[.]py" 2>/dev/null; }

# ./almond serve's wrapper ignores INT/TERM/HUP on purpose: it waits for its server, then switches power-tune off.
# It exits with its server; a leftover wrapper (its server already gone) is killed below once the servers are done.
launchers() { pgrep -f "$R/tools/cli/launch[.]py" 2>/dev/null; }

# the engine and the image encoder, matched by executable + cwd rather than by process name: the engine's threads can
# rename the process, and a zombie child has no exe (it is reaped when its server dies).
workers() {
    for p in $(pgrep -f 'almond-(engine|vision)' 2>/dev/null); do
        case "$(readlink "/proc/$p/exe" 2>/dev/null)" in
            "$R/build-hip-fused/almond-engine"|"$R/build-vision/build/bin/almond-vision"|"$R/build-vision-hy4/bin/almond-vision")
                [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$R" ] && echo "$p" ;;
        esac
    done
}

pids="$(servers) $(workers)"
launch="$(launchers)"
if [ -z "$pids" ] && [ -z "$launch" ]; then
    echo "stop.sh: no Almond process is running"
    exit 0
fi
for p in $pids; do kill -INT "$p" 2>/dev/null; done
echo "stop.sh: stopping ($(echo $pids $launch | tr -s ' ' ','))"
i=0
while [ "$i" -lt "$TIMEOUT" ]; do
    [ -z "$(servers)" ] && [ -z "$(workers)" ] && break
    sleep 1
    i=$((i + 1))
done
left="$(servers) $(workers)"
if [ -z "$(echo $left)" ]; then
    for p in $launch; do kill -KILL "$p" 2>/dev/null; done
    echo "stop.sh: stopped in ${i} s"
    exit 0
fi
if [ "$FORCE" = 1 ]; then
    for p in $left $(launchers); do kill -KILL "$p" 2>/dev/null; done
    echo "stop.sh: still running after ${TIMEOUT} s: killed $(echo $left | tr -s ' ' ',')"
    # ./almond serve may have been killed with them: switch power-tune off here if it is installed
    [ -x /usr/local/sbin/almond-power-tune ] && sudo -n /usr/local/sbin/almond-power-tune off >/dev/null 2>&1 &&
        echo "stop.sh: power-tune off"
    exit 0
fi
echo "stop.sh: still running after ${TIMEOUT} s ($(echo $left | tr -s ' ' ',')); ./stop.sh --force kills it" >&2
exit 1
