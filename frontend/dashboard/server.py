import sqlite3
import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import os
from datetime import date, timedelta

DB_PATH = 'reefer.db'
_analytics_cache = None  # computed once per server run

class DashboardAPIHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path == '/api/containers':
            self.handle_get_containers(parse_qs(parsed_url.query))
            return
        if parsed_url.path == '/api/data':
            q = parse_qs(parsed_url.query)
            if 'uuid' in q:
                self.handle_get_data(q['uuid'][0])
                return
            else:
                self.send_error(400, "Missing 'uuid' parameter")
                return
        if parsed_url.path == '/api/overview-analytics':
            self.handle_overview_analytics()
            return
        super().do_GET()

    def _send_json(self, data):
        payload = json.dumps(data, default=str)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(payload.encode())

    # ------------------------------------------------------------------
    # GET /api/containers?limit=50&offset=0&sort=total_connected_hours&dir=DESC&q=
    # ------------------------------------------------------------------
    def handle_get_containers(self, params):
        try:
            limit  = int(params.get('limit',  ['50'])[0])
            offset = int(params.get('offset', ['0'])[0])
            sort   = params.get('sort', ['total_connected_hours'])[0]
            dir_   = 'DESC' if params.get('dir', ['DESC'])[0].upper() == 'DESC' else 'ASC'
            query  = params.get('q', [''])[0].strip()

            ALLOWED_SORT = {'uuid': 'container_uuid', 'num_visits': 'num_visits',
                            'total_connected_hours': 'total_connected_hours',
                            'avg_visit_hours': 'avg_visit_hours'}
            sort_col = ALLOWED_SORT.get(sort, 'total_connected_hours')

            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            where = ''
            args  = []
            if query:
                where = 'WHERE container_uuid LIKE ?'
                args.append(f'%{query}%')

            cur.execute(f"SELECT COUNT(*) FROM container_stats {where}", args)
            total = cur.fetchone()[0]

            cur.execute(f"""
                SELECT container_uuid, num_visits, total_connected_hours,
                       avg_visit_hours, last_visit_start, last_visit_end
                FROM container_stats
                {where}
                ORDER BY {sort_col} {dir_}
                LIMIT ? OFFSET ?
            """, args + [limit, offset])

            containers = [dict(r) for r in cur.fetchall()]
            conn.close()

            # Normalize key name for frontend
            for c in containers:
                c['uuid'] = c.pop('container_uuid')

            self._send_json({'containers': containers, 'total': total, 'limit': limit, 'offset': offset})
        except Exception as e:
            self.send_error(500, str(e))

    # ------------------------------------------------------------------
    # GET /api/overview-analytics  (cached)
    # ------------------------------------------------------------------
    def handle_overview_analytics(self):
        global _analytics_cache
        if _analytics_cache is not None:
            self._send_json(_analytics_cache)
            return
        try:
            _analytics_cache = compute_analytics()
            self._send_json(_analytics_cache)
        except Exception as e:
            self.send_error(500, str(e))

    # ------------------------------------------------------------------
    # GET /api/data?uuid=...
    # ------------------------------------------------------------------
    def handle_get_data(self, uuid):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("""
                SELECT EventTime, container_visit_uuid,
                       AvPowerCons, TtlEnergyConsHour, TtlEnergyCons,
                       TemperatureSetPoint, TemperatureAmbient,
                       TemperatureReturn, RemperatureSupply,
                       HardwareType, ContainerSize, stack_tier
                FROM events
                WHERE container_uuid = ?
                ORDER BY EventTime ASC
            """, (uuid,))

            timeline = []
            for row in cur.fetchall():
                timeline.append({
                    "time":           row["EventTime"],
                    "visit_uuid":     row["container_visit_uuid"],
                    "power_kw":       round((row["AvPowerCons"] or 0) / 1000.0, 2),
                    "energy_hour":    row["TtlEnergyConsHour"],
                    "energy_total":   row["TtlEnergyCons"],
                    "setpoint":       row["TemperatureSetPoint"],
                    "ambient":        row["TemperatureAmbient"],
                    "temp_return":    row["TemperatureReturn"],
                    "temp_supply":    row["RemperatureSupply"],
                    "hardware_type":  row["HardwareType"],
                    "container_size": row["ContainerSize"],
                    "stack_tier":     row["stack_tier"],
                })

            cur.execute("""
                SELECT container_visit_uuid, visit_start, visit_end, duration_hours,
                       reading_count, hardware_type, container_size, avg_power_kw
                FROM visit_stats WHERE container_uuid = ? ORDER BY visit_start ASC
            """, (uuid,))
            visits = [{"visit_uuid": r["container_visit_uuid"], "visit_start": r["visit_start"],
                       "visit_end": r["visit_end"], "duration_hours": r["duration_hours"],
                       "reading_count": r["reading_count"], "hardware_type": r["hardware_type"],
                       "container_size": r["container_size"], "avg_power_kw": r["avg_power_kw"]}
                      for r in cur.fetchall()]

            cur.execute("""
                SELECT num_visits, total_connected_hours, avg_visit_hours,
                       last_visit_start, last_visit_end
                FROM container_stats WHERE container_uuid = ?
            """, (uuid,))
            stats = cur.fetchone()
            conn.close()

            self._send_json({
                "timeline": timeline, "visits": visits,
                "num_visits": stats["num_visits"] if stats else 0,
                "total_connected_hours": stats["total_connected_hours"] if stats else 0,
                "avg_visit_hours": stats["avg_visit_hours"] if stats else 0,
                "last_visit_start": stats["last_visit_start"] if stats else None,
                "last_visit_end": stats["last_visit_end"] if stats else None,
            })
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        pass  # silence per-request noise


# ------------------------------------------------------------------
# Analytics computation (runs once, cached)
# ------------------------------------------------------------------
def compute_analytics():
    print("Computing fleet analytics (one-time)...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    result = {}

    # 1. Active containers per day
    cur = conn.cursor()
    cur.execute("""
        SELECT date(EventTime) as day, COUNT(DISTINCT container_visit_uuid) as active
        FROM events
        GROUP BY day
        ORDER BY day
    """)
    rows = cur.fetchall()
    result['active_per_day'] = {
        'dates':  [r['day'] for r in rows],
        'counts': [r['active'] for r in rows]
    }

    # 2. Hardware type distribution
    cur.execute("""
        SELECT hardware_type, COUNT(*) as cnt,
               ROUND(AVG(avg_power_kw), 2) as avg_kw
        FROM visit_stats WHERE hardware_type IS NOT NULL
        GROUP BY hardware_type ORDER BY cnt DESC
    """)
    result['hardware_types'] = [dict(r) for r in cur.fetchall()]

    # 3. Visit duration histogram
    bins = [(0,12,'<12h'),(12,24,'12–24h'),(24,72,'1–3d'),(72,168,'3–7d'),(168,336,'1–2w'),(336,720,'2–4w'),(720,9999,'>4w')]
    bin_counts = {lbl: 0 for *_, lbl in bins}
    cur.execute("SELECT duration_hours FROM visit_stats WHERE duration_hours IS NOT NULL")
    for row in cur.fetchall():
        h = row['duration_hours']
        for lo, hi, lbl in bins:
            if lo <= h < hi:
                bin_counts[lbl] += 1
                break
    result['duration_hist'] = {
        'labels': [lbl for *_, lbl in bins],
        'counts': [bin_counts[lbl] for *_, lbl in bins]
    }

    # 4. Monthly energy totals (MWh)
    cur.execute("""
        SELECT strftime('%Y-%m', EventTime) as month,
               ROUND(SUM(AvPowerCons) / 1000000.0, 2) as total_mwh
        FROM events WHERE EventTime IS NOT NULL
        GROUP BY month ORDER BY month
    """)
    result['monthly_energy'] = [dict(r) for r in cur.fetchall()]

    # 5. SetPoint distribution
    cur.execute("""
        SELECT CAST(ROUND(TemperatureSetPoint / 5.0) * 5 AS INTEGER) as sp_bin,
               COUNT(*) as cnt
        FROM events
        WHERE TemperatureSetPoint IS NOT NULL
          AND TemperatureSetPoint > -50 AND TemperatureSetPoint < 35
        GROUP BY sp_bin ORDER BY sp_bin
    """)
    result['setpoint_dist'] = [dict(r) for r in cur.fetchall()]

    # 6. Container size vs avg power
    cur.execute("""
        SELECT CAST(ContainerSize AS INTEGER) as size, COUNT(*) as cnt,
               ROUND(AVG(AvPowerCons) / 1000.0, 2) as avg_power_kw
        FROM events WHERE ContainerSize IS NOT NULL
        GROUP BY size ORDER BY size
    """)
    result['container_sizes'] = [dict(r) for r in cur.fetchall()]

    # 7. Hour × Day-of-Week heatmap (active container visits per slot)
    cur.execute("""
        SELECT CAST(strftime('%H', EventTime) AS INTEGER) as hour,
               CAST(strftime('%w', EventTime) AS INTEGER) as dow,
               COUNT(DISTINCT container_visit_uuid) as count
        FROM events
        WHERE EventTime IS NOT NULL
        GROUP BY hour, dow
        ORDER BY dow, hour
    """)
    result['hourly_heatmap'] = [dict(r) for r in cur.fetchall()]

    conn.close()
    print("Analytics ready.")
    return result


if __name__ == '__main__':
    PORT = 8080
    if not os.path.exists(DB_PATH):
        print("ERROR: reefer.db not found! Run build_database.py first.")
        exit(1)

    print(f"===========================================================")
    print(f"Eurogate API Data-Server running on port {PORT}")
    print(f"Navigate to http://localhost:{PORT}/hamburg.html")
    print(f"===========================================================")

    import socketserver
    socketserver.TCPServer.allow_reuse_address = True
    httpd = HTTPServer(('0.0.0.0', PORT), DashboardAPIHandler)
    httpd.serve_forever()
