#!/usr/bin/env bash
# Hit every endpoint and report what works. Usage: scripts/smoke_test.sh [base_url]
BASE="${1:-http://localhost:8000}"

hit () {  # hit <label> <path>
  printf '\n\033[1m%s\033[0m\n  GET %s\n' "$1" "$2"
  code=$(curl -s -o /tmp/_lux_body -w '%{http_code}' "$BASE$2")
  if [ "$code" = 200 ]; then printf '  \033[32mHTTP %s\033[0m\n' "$code"
  else printf '  \033[31mHTTP %s\033[0m\n' "$code"; fi
  head -c 400 /tmp/_lux_body; echo
}

echo "testing $BASE"
hit "1. health"                "/health"
hit "2. counter list (1058 series)" "/counters"
hit "3. forecast - busiest counter (1410, cars)" \
    "/forecast?poste_id=1410&direction=1&vehicule=V&date=2025-03-12"
hit "4. forecast - quiet truck counter (26, trucks)" \
    "/forecast?poste_id=26&direction=1&vehicule=C&date=2025-01-09"
hit "5. forecast - public holiday (New Year)" \
    "/forecast?poste_id=1410&direction=1&vehicule=V&date=2025-01-01"
hit "6. error case - unknown counter (expect 404)" \
    "/forecast?poste_id=999999&direction=1&vehicule=V&date=2025-03-12"
hit "7. error case - bad vehicule (expect 422)" \
    "/forecast?poste_id=1410&direction=1&vehicule=X&date=2025-03-12"
hit "8. error case - bad date (expect 422, currently 404)" \
    "/forecast?poste_id=1410&direction=1&vehicule=V&date=not-a-date"

printf '\n\033[1mdaily totals for one counter, whole week\033[0m\n'
for d in 2025-03-10 2025-03-11 2025-03-12 2025-03-13 2025-03-14 2025-03-15 2025-03-16; do
  t=$(curl -s "$BASE/forecast?poste_id=1410&direction=1&vehicule=V&date=$d" \
      | python3 -c 'import sys,json
d = json.load(sys.stdin)
print("{:>9,}  holiday={}".format(d["daily_total"], d["is_holiday_period"]))' 2>/dev/null)
  echo "  $d  $t"
done
echo
echo "interactive docs: $BASE/docs"
