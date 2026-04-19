#!/bin/sh
# presenced — waybar custom-module status script.
#
# Emits a single JSON line on stdout. presenced does not export its FSM
# state over a socket/dbus, so this reads the last "frm -> to" transition
# out of the systemd journal (cheap: journalctl -n with --no-pager).
#
# States: present / away_grace / away / off / unknown
#
# Install: point a waybar "custom/presenced" module at this script with
# return-type=json. See project-b/contrib/waybar/config.jsonc.

if ! systemctl --user is-active --quiet presenced.service; then
    printf '{"text":"\u25cb","alt":"off","tooltip":"presenced: off","class":"off"}\n'
    exit 0
fi

last=$(journalctl --user -u presenced.service -n 200 --no-pager -o cat 2>/dev/null \
    | grep -oE '-> (present|away_grace|away)' \
    | tail -1 \
    | awk '{print $2}')

case "$last" in
    present)
        printf '{"text":"\u25cf","alt":"present","tooltip":"presenced: PRESENT","class":"present"}\n'
        ;;
    away_grace)
        printf '{"text":"\u25d0","alt":"grace","tooltip":"presenced: AWAY (grace)","class":"grace"}\n'
        ;;
    away)
        printf '{"text":"\u25cb","alt":"away","tooltip":"presenced: AWAY","class":"away"}\n'
        ;;
    *)
        printf '{"text":"?","alt":"unknown","tooltip":"presenced: active, no transitions yet","class":"unknown"}\n'
        ;;
esac
