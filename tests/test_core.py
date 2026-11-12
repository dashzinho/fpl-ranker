"""Run: python -m unittest discover -s tests   (from the project folder)"""
import json, os, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import fpl_data as fd, fpl_extra as fx, fpl_model as fm, fpl_news as fn   # noqa: E402


class BanProbs(unittest.TestCase):
    seq = [(11 + i, None, 1.0) for i in range(6)]

    def test_four_yellows_is_risky(self):
        ban, risk = fm.ban_probs(4, 0.25, 90, self.seq)
        self.assertGreater(risk, 0.3)
        self.assertEqual(ban[0], 0.0)                  # cannot be banned for the very next match without a card first
        self.assertGreater(max(ban), 0.1)

    def test_clean_record_no_risk(self):
        ban, risk = fm.ban_probs(0, 0.25, 90, self.seq)
        self.assertEqual(risk, 0.0); self.assertEqual(sum(ban), 0.0)

    def test_after_cutoff_gameweek_no_ban(self):
        late = [(30 + i, None, 1.0) for i in range(4)]    # 5-yellow rule ends at GW19; 10-yellow rule not in range for y=4
        ban, risk = fm.ban_probs(4, 0.25, 90, late)
        self.assertEqual(risk, 0.0)

    def test_probabilities_stay_valid(self):
        for y in range(0, 12):
            ban, risk = fm.ban_probs(y, 0.4, 90, self.seq)
            self.assertTrue(all(0 <= b <= 1 for b in ban)); self.assertTrue(0 <= risk <= 1)

    def test_unavailable_player_cannot_get_cards(self):
        ban, risk = fm.ban_probs(4, 0.4, 90, [(11 + i, None, 0.0) for i in range(5)])
        self.assertEqual(risk, 0.0)


class FreeTransfers(unittest.TestCase):
    def hist(self, rows):
        return [{"event": g, "event_transfers": n, "event_transfers_cost": c} for g, n, c in rows]

    def test_unused_transfers_bank(self):
        self.assertEqual(fd._free_transfers(self.hist([(2, 0, 0), (3, 0, 0), (4, 0, 0)]), 5), 4)

    def test_cap_of_five(self):
        self.assertEqual(fd._free_transfers(self.hist([(g, 0, 0) for g in range(2, 12)]), 12), 5)

    def test_sell_price_rule(self):
        self.assertEqual(fd._sell_price(61, 60, True), 60)         # rise of 0.1: no profit yet
        self.assertEqual(fd._sell_price(63, 60, True), 61)         # rise of 0.3: sell at buy + 0.1
        self.assertEqual(fd._sell_price(57, 60, True), 57)         # falls are fully lost
        self.assertEqual(fd._sell_price(57, None, False), 57)


class Snapshot(unittest.TestCase):
    def test_conversion(self):
        raw = {"picks": [{"element": i, "position": i, "selling_price": 50, "purchase_price": 49, "multiplier": 1,
                          "is_captain": i == 1, "is_vice_captain": i == 2} for i in range(1, 16)],
               "chips": [{"name": "freehit", "status_for_entry": "available", "played_by_entry": [], "stop_event": 19},
                         {"name": "wildcard", "status_for_entry": "played", "played_by_entry": [3], "stop_event": 19}],
               "transfers": {"cost": 4, "limit": 2, "made": 1, "bank": 7, "value": 1037, "status": "cost"}}
        sn = fd.snapshot_from_raw(1, raw, 11)
        self.assertEqual(sn["transfers"]["bank"], 0.7); self.assertEqual(sn["picks"][0]["selling_price"], 5.0)
        self.assertEqual(sn["chip_window_ends_gw"], 19)
        self.assertEqual([c["name"] for c in sn["chips"] if c["status"] == "available"], ["freehit"])


class AIValidation(unittest.TestCase):
    boot = {"elements": [{"id": i, "web_name": f"P{i}"} for i in range(1, 30)], "events": [{"id": 11, "is_next": True}]}

    def load(self, payload):
        with tempfile.TemporaryDirectory() as d:
            old = fx.INSIGHTS_FILE
            fx.INSIGHTS_FILE = os.path.join(d, "x.json")
            json.dump(payload, open(fx.INSIGHTS_FILE, "w"))
            try:
                return fx.load_insights(self.boot, {"n1", "n2"})
            finally:
                fx.INSIGHTS_FILE = old

    def test_valid_adjustment_kept_and_clamped(self):
        r = self.load({"adjustments": [{"player_id": 3, "multiplier": 99, "until_gw": 12, "reason": "x" * 900, "evidence": ["n1"]}]})
        a = r["adjustments"][0]
        self.assertEqual(a["mult"], 1.2); self.assertLessEqual(len(a["reason"]), 400)

    def test_rejects_unknown_player_missing_evidence_and_made_up_refs(self):
        r = self.load({"adjustments": [{"player_id": 999, "multiplier": 0.5, "evidence": ["n1"]},
                                       {"player_id": 3, "multiplier": 0.5, "evidence": []},
                                       {"player_id": 4, "multiplier": 0.5, "evidence": ["n77"]},
                                       {"player_id": 5, "multiplier": "abc", "evidence": ["n1"]}]})
        self.assertEqual(r["adjustments"], []); self.assertEqual(len(r["dropped"]), 4)

    def test_fpl_status_evidence_allowed_and_floor(self):
        r = self.load({"adjustments": [{"player_id": 3, "multiplier": 0.0, "evidence": ["fpl_status"]}]})
        self.assertEqual(r["adjustments"][0]["mult"], 0.2)

    def test_hostile_text_is_stripped_of_control_chars_and_bad_transfers_dropped(self):
        r = self.load({"summary": "ok\x00\x1b[31m", "transfers": [{"out": 1, "in": 1}, {"out": 1, "in": 500}, {"out": 1, "in": 2}],
                       "captain": {"player_id": 2, "alternatives": [3, 4, 5, 6, 400]}, "chips": {"recommendation": "rm -rf", "gw": "x"}})
        self.assertNotIn("\x00", r["summary"]); self.assertNotIn("\x1b", r["summary"])
        self.assertEqual(len(r["transfers"]), 1); self.assertEqual(r["chips"]["rec"], "none"); self.assertIsNone(r["chips"]["gw"])
        self.assertLessEqual(len(r["captain"]["alts"]), 3)

    def test_garbage_file_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            old = fx.INSIGHTS_FILE; fx.INSIGHTS_FILE = os.path.join(d, "x.json"); open(fx.INSIGHTS_FILE, "w").write("not json")
            try: self.assertIsNone(fx.load_insights(self.boot, set()))
            finally: fx.INSIGHTS_FILE = old


class NewsFilter(unittest.TestCase):
    def test_excludes_other_sports_and_womens_football(self):
        for t in ["Women's Super League: Arsenal beat Chelsea", "Lionesses squad named", "Grand Slam of Darts draw", "Chelsea Women sign striker",
                  "https://www.bbc.co.uk/sport/football/womens/123", "Formula 1 qualifying report"]:
            self.assertTrue(fn.EXCLUDE.search(t), t)

    def test_keeps_mens_premier_league(self):
        for t in ["Haaland ruled out with knock", "Arsenal injury news: Saka doubt", "Salah returns to training", "Premier League team news"]:
            self.assertFalse(fn.EXCLUDE.search(t), t)


if __name__ == "__main__":
    unittest.main()
