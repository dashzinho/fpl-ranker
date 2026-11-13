"""Run: python -m unittest discover -s tests   (from the project folder)"""
import json, os, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import backtest as bt, fpl_data as fd, fpl_extra as fx, fpl_model as fm, fpl_news as fn   # noqa: E402


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


class Weights(unittest.TestCase):
    def test_save_merges_and_validates(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"fpl_team_id": 7}, open(os.path.join(d, "prefs.local.json"), "w"))
            w = fx.save_weights(d, {"wo": 80, "wp": "0", "wv": 20})
            self.assertEqual(w, {"outlook": 80, "perf": 0, "value": 20})
            saved = json.load(open(os.path.join(d, "prefs.local.json")))
            self.assertEqual(saved["fpl_team_id"], 7); self.assertEqual(saved["weights"]["value"], 20)

    def test_rejects_bad_weights_and_leaves_file_alone(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in ({"wo": 0, "wp": 0, "wv": 0}, {"wo": 101, "wp": 0, "wv": 0}, {"wo": -1, "wp": 5, "wv": 5},
                        {"wo": "x", "wp": 1, "wv": 1}, {"wo": 1, "wp": 1}, None, {"wo": 1.5e9, "wp": 1, "wv": 1}):
                with self.assertRaises(ValueError): fx.save_weights(d, bad)
            self.assertFalse(os.path.exists(os.path.join(d, "prefs.local.json")))


class BacktestMaths(unittest.TestCase):
    def test_percentile_matches_the_page(self):
        out = bt._pct([10, 20, 30, 40], [True, True, True, True])
        self.assertEqual(list(out), [0.0, 100 / 3, 200 / 3, 100.0])

    def test_ineligible_players_score_zero_and_ties_share_rank(self):
        out = bt._pct([5, 5, 9, 1], [True, True, True, False])
        self.assertEqual(out[3], 0.0); self.assertEqual(out[0], out[1]); self.assertGreater(out[2], out[0])

    def test_mix_grid_is_complete(self):
        self.assertTrue(all(a + b + c == 100 for a, b, c in bt.MIX)); self.assertEqual(len(bt.MIX), 231)

    def test_cut_stats_use_only_given_rows(self):
        p = {}
        bt._cut_stats(p, [{"minutes": 90, "total_points": 6, "bonus": 1, "expected_goal_involvements": "0.5"},
                          {"minutes": 90, "total_points": 2, "bonus": 0, "expected_goal_involvements": "0.5"}])
        self.assertEqual((p["mins"], p["pts"], round(p["p90"], 2), round(p["xgi90"], 2), round(p["bn90"], 2)), (180, 8, 4.0, 0.5, 0.5))


class WeightSets(unittest.TestCase):
    def test_team_sets_contain_the_default_and_presets(self):
        self.assertIn(tuple(bt.TM_DEFAULT[k] for k in bt.TM_KEYS), bt.TM_SETS)
        self.assertEqual(len(set(bt.TM_SETS)), len(bt.TM_SETS))
        for nm, x, _ in bt.PRESETS:
            self.assertIn(tuple(x[:4]), bt.TM_SETS, nm)
            self.assertIn(tuple(x[4:]), bt.MIX, nm)
        self.assertEqual(bt._find({}, (0.65, 12, 5, 0.10, 60, 25, 15)), (0, bt.MIX.index((60, 25, 15))))
        self.assertIsNone(bt._find({}, (0.5, 12, 5, 0.10, 60, 25, 15)))

    def test_overlap_aware_noise_is_never_below_the_plain_one(self):
        import numpy as np
        rng = np.random.default_rng(1)
        base = rng.normal(size=(50, 12))
        smooth = base[:, :-2] + base[:, 1:-1] + base[:, 2:]                                  # overlapping windows
        plain = smooth.std(axis=1, ddof=1) / np.sqrt(smooth.shape[1])
        self.assertTrue((bt._se_hac(smooth) >= plain - 1e-12).all())
        self.assertGreater(float(np.mean(bt._se_hac(smooth) / plain)), 1.0)                  # positive autocorrelation widens it

    def test_walk_forward_only_uses_earlier_gameweeks(self):
        import numpy as np
        nS, nM, last = len(bt.TM_SETS), len(bt.MIX), 10
        gws = list(range(1, last + 1)); pgws = list(range(4, last + 1))
        rng = np.random.default_rng(0)
        T = {k: rng.normal(size=(nS, len(gws))) for k in ("gd", "xgd", "ll")}
        P = rng.normal(size=(nS, nM, len(pgws)))
        wf = bt._walkforward(T, P, gws, pgws)
        self.assertEqual(wf["gws"], [6, 7, 8, 9, 10])
        # plant a set that is great ONLY in the last gameweek: it must not be picked for any earlier test week
        T2 = {k: v.copy() for k, v in T.items()}; P2 = P.copy()
        for k in T2:
            T2[k][3, -1] = 99.0
        P2[3, :, -1] = 99.0
        wf2 = bt._walkforward(T2, P2, gws, pgws)
        self.assertEqual(wf2["team"]["picks"][:-1], wf["team"]["picks"][:-1])

    def test_official_fdr_as_of_uses_the_value_seen_then(self):
        import fpl_data as fd
        f = {"id": 7, "team_h_difficulty": 4, "team_a_difficulty": 2}
        self.assertEqual(fd.fdr_as_of({}, f, 3), (4, 2))                                      # no history: today's value
        h = {"7": [[1, 3, 3], [8, 4, 2]]}
        self.assertEqual(fd.fdr_as_of(h, f, 5), (3, 3))
        self.assertEqual(fd.fdr_as_of(h, f, 9), (4, 2))

    def test_last_good_squad_is_reused_and_marked_stale(self):
        import fpl_data as fd
        old = fd.MY_CACHE
        with tempfile.TemporaryDirectory() as d:
            fd.MY_CACHE = os.path.join(d, "c.json")
            try:
                self.assertIsNone(fd.cached_my_team(1))
                fd.cache_my_team(1, {"picks": [1], "bank": 0.5})
                got = fd.cached_my_team(1)
                self.assertEqual((got["bank"], bool(got["stale"])), (0.5, True))
                self.assertIsNone(fd.cached_my_team(2))                                       # another team id: no leak
                fd.cache_my_team(1, {"error": "boom"})
                self.assertIsNone(fd.cached_my_team(1))                                       # never serve an error as a squad
            finally:
                fd.MY_CACHE = old

    def test_save_model_set_validates_merges_and_stops_auto_pick(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"fpl_team_id": 7}, open(os.path.join(d, "prefs.local.json"), "w"))
            fx.save_model_set(d, {"xg_weight": 1, "half_life_gw": 4, "prior_strength": 10, "prior_scale": 0.2, "wo": 50, "wp": 30, "wv": 20})
            cur = json.load(open(os.path.join(d, "prefs.local.json")))
            self.assertEqual((cur["fpl_team_id"], cur["half_life_gw"], cur["prior_scale"], cur["weights"]["perf"], cur["auto_pick"]), (7, 4, 0.2, 30, False))
            ok = {"xg_weight": 1, "half_life_gw": 4, "prior_strength": 1, "prior_scale": .1}
            for bad in (dict(ok, xg_weight=2), {"xg_weight": .5}, dict(ok, xg_weight="x"), dict(ok, wo=50, wp=30, wv=30)):
                with self.assertRaises(ValueError):
                    fx.save_model_set(d, bad)

    def test_xg_weight_changes_the_fit_target(self):
        boot = {"teams": [{"id": i, "strength_overall_home": 3, "strength_overall_away": 3} for i in (1, 2)]}
        fixt = [{"id": 1, "finished": True, "event": 1, "team_h": 1, "team_a": 2, "team_h_score": 0, "team_a_score": 0}]
        xg = {(1, 1): 3.0, (1, 2): 0.0}
        hi = fm.fit_teams(boot, fixt, xg, 1, {"xg_weight": 1.0, "prior_strength": 0.1})
        lo = fm.fit_teams(boot, fixt, xg, 1, {"xg_weight": 0.0, "prior_strength": 0.1})
        self.assertGreater(hi["lam"](1, 2, True), lo["lam"](1, 2, True))


class AutoPick(unittest.TestCase):
    META = {"finished": 10, "best_team": [1.0, 24, 20, 0.2]}

    def test_picks_once_per_gameweek_changes_only_the_team_model_and_can_revert(self):
        import autopick as ap
        old = ap.FILE
        with tempfile.TemporaryDirectory() as d:
            ap.FILE = os.path.join(d, "auto_pick.json")
            json.dump({"fpl_team_id": 1, "weights": {"outlook": 70, "perf": 20, "value": 10}}, open(os.path.join(d, "prefs.local.json"), "w"))
            try:
                prefs = {"auto_pick": True, "weights": {"outlook": 70, "perf": 20, "value": 10}}
                self.assertTrue(ap.apply(d, prefs, self.META, lambda *_: None))              # first run: picks
                self.assertEqual(prefs["half_life_gw"], 24)
                self.assertEqual(prefs["weights"], {"outlook": 70, "perf": 20, "value": 10})  # Score weights are never touched
                saved = json.load(open(os.path.join(d, "prefs.local.json")))
                self.assertEqual((saved["half_life_gw"], saved["weights"]["outlook"]), (24, 70))
                self.assertFalse(ap.apply(d, prefs, self.META, lambda *_: None))             # same gameweek: nothing
                p2 = {"auto_pick": True}
                m2 = dict(self.META, finished=11, best_team=[0.0, 4, 1, 0.05])
                self.assertTrue(ap.apply(d, p2, m2, lambda *_: None))                        # new gameweek: picks again
                p3 = dict(p2, auto_pick=False)
                m3 = dict(m2, finished=12, best_team=[0.65, 12, 5, 0.10])
                self.assertFalse(ap.apply(d, p3, m3, lambda *_: None))                       # off: leaves your choice alone
                ap.pick_now(d)
                self.assertTrue(ap.apply(d, p3, m3, lambda *_: None))                        # manual button works even when off
                self.assertFalse(ap.apply(d, p3, m3, lambda *_: None))                       # and only once
                st = ap.status({"auto_pick": True})
                self.assertEqual(len(st["history"]), 3)
                self.assertTrue(st["can_revert"])
                back = ap.revert(d)                                                          # undo the latest change
                self.assertEqual(back["half_life_gw"], 4)
                cur = json.load(open(os.path.join(d, "prefs.local.json")))
                self.assertEqual((cur["half_life_gw"], cur["auto_pick"]), (4, False))
                self.assertTrue(ap.enabled({}))
                self.assertFalse(ap.enabled({"auto_pick": False}))
            finally:
                ap.FILE = old

    def test_revert_never_overwrites_a_later_manual_change_or_repeats(self):
        import autopick as ap
        old = ap.FILE
        with tempfile.TemporaryDirectory() as d:
            ap.FILE = os.path.join(d, "auto_pick.json")
            json.dump({"fpl_team_id": 1}, open(os.path.join(d, "prefs.local.json"), "w"))
            try:
                prefs = {"auto_pick": True}
                self.assertTrue(ap.apply(d, prefs, self.META, lambda *_: None))
                self.assertTrue(ap.status(prefs)["can_revert"])
                fx.save_model_set(d, {"xg_weight": 0.65, "half_life_gw": 7, "prior_strength": 5, "prior_scale": 0.1})   # a manual change
                with self.assertRaises(ValueError):                                      # must not undo the pick over it
                    ap.revert(d)
                self.assertEqual(json.load(open(os.path.join(d, "prefs.local.json")))["half_life_gw"], 7)
                self.assertFalse(ap.status(json.load(open(os.path.join(d, "prefs.local.json"))))["can_revert"])
            finally:
                ap.FILE = old
        with tempfile.TemporaryDirectory() as d:
            ap.FILE = os.path.join(d, "auto_pick.json")
            json.dump({"fpl_team_id": 1}, open(os.path.join(d, "prefs.local.json"), "w"))
            try:
                ap.apply(d, {"auto_pick": True}, self.META, lambda *_: None)
                ap.revert(d)
                with self.assertRaises(ValueError):                                      # already reverted: nothing more to undo
                    ap.revert(d)
            finally:
                ap.FILE = old

    def test_revert_with_nothing_to_revert_raises(self):
        import autopick as ap
        old = ap.FILE
        with tempfile.TemporaryDirectory() as d:
            ap.FILE = os.path.join(d, "auto_pick.json")
            try:
                with self.assertRaises(ValueError):
                    ap.revert(d)
            finally:
                ap.FILE = old

    def test_sliders_save_stops_auto_pick(self):
        with tempfile.TemporaryDirectory() as d:
            fx.save_weights(d, {"wo": 50, "wp": 25, "wv": 25})
            self.assertEqual(json.load(open(os.path.join(d, "prefs.local.json")))["auto_pick"], False)


if __name__ == "__main__":
    unittest.main()
