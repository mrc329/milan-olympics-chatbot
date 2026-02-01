"""
test_milan2026_pipeline.py
==========================
unittest suite for milan2026_pipeline.py.

Run:
    python -m pytest test_milan2026_pipeline.py -v
  or:
    python test_milan2026_pipeline.py

Every test resets shared pipeline state (VECTOR_STORE, UPDATED_VECTORS,
EVENT_RESULTS_THIS_RUN, INJURIES_THIS_RUN, RUMORS_THIS_RUN) in setUp so
tests are fully isolated.

Coverage map
────────────
TestUtils                  slug, resolve_mode, freshness_metadata
TestDiscoverEntities       entity lists per mode
TestUpsertVector           insert vs update detection
TestNarratives             upsert_narrative vector shape + metadata
TestRumors                 unconfirmed / confirmed (promotion) / denied lifecycle
TestInjuries               vector shape + INJURIES_THIS_RUN cache population
TestEvents                 vector shape + EVENT_RESULTS_THIS_RUN cache population
TestAthletes               enriched vector: bio + medals + injury flag
TestIndividualUpsets       detect_upsets: fires on non-favorites, skips team events
TestCountryUpsets          detect_country_upsets: team_event / surge / shutout signals
TestSummarizeUpdates       routing of every vector prefix into correct bucket
TestMainMode               full pipeline run per mode (PRE_GAMES, LIVE_GAMES, DORMANT)
"""

import unittest
from datetime import datetime, timezone
import milan2026_pipeline as p


def _reset():
    """Wipe all shared mutable state so each test starts clean."""
    p.VECTOR_STORE.clear()
    p.UPDATED_VECTORS.clear()
    p.EVENT_RESULTS_THIS_RUN.clear()
    p.INJURIES_THIS_RUN.clear()
    p.RUMORS_THIS_RUN.clear()


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
class TestUtils(unittest.TestCase):

    def test_slug_basic(self):
        self.assertEqual(p.slug("Hello World"), "hello_world")

    def test_slug_special_chars_stripped(self):
        # ø and á are non-ASCII; regex [^a-z0-9] strips them
        self.assertEqual(p.slug("Bjørgen"), "bj_rgen")
        self.assertEqual(p.slug("Ledecká"), "ledeck")

    def test_slug_apostrophe(self):
        self.assertEqual(p.slug("Danny O'Shea"), "danny_o_shea")

    def test_slug_leading_trailing_underscores_stripped(self):
        self.assertEqual(p.slug("  --hello--  "), "hello")

    # ── resolve_mode ──
    def test_resolve_mode_pre_games(self):
        before = datetime(2026, 2, 4, 23, 59, tzinfo=timezone.utc)
        self.assertEqual(p.resolve_mode(before), "PRE_GAMES")

    def test_resolve_mode_live_games_start(self):
        start = datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(p.resolve_mode(start), "LIVE_GAMES")

    def test_resolve_mode_live_games_end(self):
        end = datetime(2026, 2, 22, 23, 59, tzinfo=timezone.utc)
        self.assertEqual(p.resolve_mode(end), "LIVE_GAMES")

    def test_resolve_mode_dormant(self):
        after = datetime(2026, 2, 23, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(p.resolve_mode(after), "DORMANT")

    # ── freshness_metadata ──
    def test_freshness_metadata_keys(self):
        meta = p.freshness_metadata("wikipedia", "high")
        self.assertIn("source", meta)
        self.assertIn("volatility", meta)
        self.assertIn("last_fetched_utc", meta)
        self.assertEqual(meta["source"], "wikipedia")
        self.assertEqual(meta["volatility"], "high")
        self.assertTrue(meta["last_fetched_utc"].endswith("Z"))


# ─────────────────────────────────────────────
# DISCOVER ENTITIES
# ─────────────────────────────────────────────
class TestDiscoverEntities(unittest.TestCase):

    def test_pre_games_has_no_events(self):
        entities = p.discover_entities("PRE_GAMES")
        self.assertEqual(entities["events"], [])

    def test_live_games_has_events(self):
        entities = p.discover_entities("LIVE_GAMES")
        self.assertEqual(len(entities["events"]), 6)

    def test_athletes_always_present(self):
        for mode in ("PRE_GAMES", "LIVE_GAMES"):
            entities = p.discover_entities(mode)
            self.assertEqual(len(entities["athletes"]), 10)

    def test_narratives_always_present(self):
        entities = p.discover_entities("PRE_GAMES")
        self.assertEqual(len(entities["narratives"]), 3)

    def test_rumors_present(self):
        entities = p.discover_entities("PRE_GAMES")
        self.assertEqual(len(entities["rumors"]), 1)
        self.assertEqual(entities["rumors"][0]["id"], "bocelli_opening")

    def test_injuries_present(self):
        entities = p.discover_entities("PRE_GAMES")
        self.assertEqual(len(entities["injuries"]), 1)
        self.assertEqual(entities["injuries"][0]["athlete"], "Mikaela Shiffrin")


# ─────────────────────────────────────────────
# UPSERT VECTOR
# ─────────────────────────────────────────────
class TestUpsertVector(unittest.TestCase):

    def setUp(self):
        _reset()

    def test_first_write_is_inserted(self):
        action = p.upsert_vector("test::a", "text", {"doc_type": "test"})
        self.assertEqual(action, "inserted")

    def test_second_write_is_updated(self):
        p.upsert_vector("test::a", "text1", {})
        action = p.upsert_vector("test::a", "text2", {})
        self.assertEqual(action, "updated")
        self.assertEqual(p.VECTOR_STORE["test::a"]["text"], "text2")


# ─────────────────────────────────────────────
# NARRATIVES
# ─────────────────────────────────────────────
class TestNarratives(unittest.TestCase):

    def setUp(self):
        _reset()

    def test_upsert_narrative_vector_id(self):
        p.upsert_narrative("Opening ceremony", "Some text")
        self.assertIn("page::opening_ceremony", p.VECTOR_STORE)

    def test_upsert_narrative_metadata(self):
        p.upsert_narrative("Opening ceremony", "Some text")
        meta = p.VECTOR_STORE["page::opening_ceremony"]["metadata"]
        self.assertEqual(meta["doc_type"], "narrative")
        self.assertEqual(meta["title"], "Opening ceremony")


# ─────────────────────────────────────────────
# RUMORS — three lifecycle states
# ─────────────────────────────────────────────
class TestRumors(unittest.TestCase):

    def setUp(self):
        _reset()

    def _base_rumor(self, status="unconfirmed", confidence=0.75):
        return {
            "id": "bocelli_opening",
            "headline": "Bocelli rumored",
            "detail": "Detail text.",
            "confidence": confidence,
            "source": "Italian press",
            "related_entity": "Opening ceremony",
            "status": status,
        }

    def test_unconfirmed_writes_rumor_vector(self):
        p.upsert_rumor(self._base_rumor())
        self.assertIn("rumor::bocelli_opening", p.VECTOR_STORE)
        meta = p.VECTOR_STORE["rumor::bocelli_opening"]["metadata"]
        self.assertEqual(meta["doc_type"], "rumor")
        self.assertEqual(meta["status"], "unconfirmed")
        self.assertEqual(meta["conf_label"], "high")   # 0.75 ≥ 0.7

    def test_unconfirmed_low_confidence_label(self):
        p.upsert_rumor(self._base_rumor(confidence=0.3))
        meta = p.VECTOR_STORE["rumor::bocelli_opening"]["metadata"]
        self.assertEqual(meta["conf_label"], "low")

    def test_unconfirmed_moderate_confidence_label(self):
        p.upsert_rumor(self._base_rumor(confidence=0.5))
        meta = p.VECTOR_STORE["rumor::bocelli_opening"]["metadata"]
        self.assertEqual(meta["conf_label"], "moderate")

    def test_confirmed_promotes_to_narrative(self):
        p.upsert_rumor(self._base_rumor(status="confirmed"))
        # Should NOT write a rumor:: vector
        self.assertNotIn("rumor::bocelli_opening", p.VECTOR_STORE)
        # Should write a page:: vector for the related narrative
        self.assertIn("page::opening_ceremony", p.VECTOR_STORE)
        text = p.VECTOR_STORE["page::opening_ceremony"]["text"]
        self.assertIn("CONFIRMED", text)

    def test_denied_writes_nothing(self):
        p.upsert_rumor(self._base_rumor(status="denied"))
        self.assertEqual(len(p.VECTOR_STORE), 0)


# ─────────────────────────────────────────────
# INJURIES
# ─────────────────────────────────────────────
class TestInjuries(unittest.TestCase):

    def setUp(self):
        _reset()

    def _shiffrin_injury(self, severity="moderate"):
        return {
            "athlete": "Mikaela Shiffrin",
            "condition": "Left ankle sprain.",
            "severity": severity,
            "status": "training with modifications",
            "event_impact": ["Women's downhill alpine skiing"],
            "source": "USSA",
        }

    def test_writes_injury_vector(self):
        p.upsert_injury(self._shiffrin_injury())
        self.assertIn("injury::mikaela_shiffrin", p.VECTOR_STORE)

    def test_metadata_fields(self):
        p.upsert_injury(self._shiffrin_injury())
        meta = p.VECTOR_STORE["injury::mikaela_shiffrin"]["metadata"]
        self.assertEqual(meta["doc_type"], "injury")
        self.assertEqual(meta["severity"], "moderate")
        self.assertEqual(meta["athlete"], "Mikaela Shiffrin")

    def test_populates_injuries_cache(self):
        p.upsert_injury(self._shiffrin_injury())
        self.assertIn("mikaela_shiffrin", p.INJURIES_THIS_RUN)
        self.assertEqual(p.INJURIES_THIS_RUN["mikaela_shiffrin"]["severity"], "moderate")


# ─────────────────────────────────────────────
# EVENTS
# ─────────────────────────────────────────────
class TestEvents(unittest.TestCase):

    def setUp(self):
        _reset()

    def _downhill_medalists(self):
        return [
            {"rank": 1, "name": "Sara Hector",      "country": "SWE"},
            {"rank": 2, "name": "Mikaela Shiffrin", "country": "USA"},
            {"rank": 3, "name": "Ester Ledecká",    "country": "CZE"},
        ]

    def test_writes_event_vector(self):
        p.upsert_event("Women's downhill alpine skiing", self._downhill_medalists())
        self.assertIn("event::women_s_downhill_alpine_skiing", p.VECTOR_STORE)

    def test_metadata_includes_medalists(self):
        medalists = self._downhill_medalists()
        p.upsert_event("Women's downhill alpine skiing", medalists)
        meta = p.VECTOR_STORE["event::women_s_downhill_alpine_skiing"]["metadata"]
        self.assertEqual(meta["doc_type"], "event_result")
        self.assertEqual(len(meta["medalists"]), 3)

    def test_populates_event_results_cache(self):
        medalists = self._downhill_medalists()
        p.upsert_event("Women's downhill alpine skiing", medalists)
        self.assertIn("Women's downhill alpine skiing", p.EVENT_RESULTS_THIS_RUN)
        self.assertEqual(len(p.EVENT_RESULTS_THIS_RUN["Women's downhill alpine skiing"]), 3)


# ─────────────────────────────────────────────
# ATHLETES (enriched)
# ─────────────────────────────────────────────
class TestAthletes(unittest.TestCase):

    def setUp(self):
        _reset()

    def test_athlete_vector_created(self):
        p.upsert_athlete({"name": "Yuzuru Hanyu", "events": ["Men's figure skating free skate"], "favorite": True})
        self.assertIn("athlete::yuzuru_hanyu", p.VECTOR_STORE)

    def test_athlete_has_no_medals_pre_games(self):
        # EVENT_RESULTS_THIS_RUN is empty → "None yet"
        p.upsert_athlete({"name": "Yuzuru Hanyu", "events": [], "favorite": True})
        text = p.VECTOR_STORE["athlete::yuzuru_hanyu"]["text"]
        self.assertIn("None yet", text)

    def test_athlete_picks_up_medal(self):
        # Seed an event result where Hanyu gets gold
        p.EVENT_RESULTS_THIS_RUN["Men's figure skating free skate"] = [
            {"rank": 1, "name": "Yuzuru Hanyu",   "country": "JPN"},
            {"rank": 2, "name": "Kagiyama Kaito", "country": "JPN"},
            {"rank": 3, "name": "Shoma Uno",      "country": "JPN"},
        ]
        p.upsert_athlete({"name": "Yuzuru Hanyu", "events": ["Men's figure skating free skate"], "favorite": True})
        text = p.VECTOR_STORE["athlete::yuzuru_hanyu"]["text"]
        self.assertIn("Gold", text)

    def test_athlete_picks_up_injury_flag(self):
        # Seed injury cache
        p.INJURIES_THIS_RUN["mikaela_shiffrin"] = {
            "athlete": "Mikaela Shiffrin",
            "condition": "Ankle sprain.",
            "severity": "moderate",
            "status": "training with modifications",
            "event_impact": ["Women's downhill alpine skiing"],
        }
        p.upsert_athlete({"name": "Mikaela Shiffrin", "events": ["Women's downhill alpine skiing"], "favorite": True})
        text = p.VECTOR_STORE["athlete::mikaela_shiffrin"]["text"]
        self.assertIn("INJURY FLAG", text)
        self.assertIn("MODERATE", text)

    def test_athlete_metadata_injury_risk(self):
        p.INJURIES_THIS_RUN["mikaela_shiffrin"] = {
            "athlete": "Mikaela Shiffrin",
            "condition": "Ankle sprain.",
            "severity": "high",
            "status": "doubtful",
            "event_impact": [],
        }
        p.upsert_athlete({"name": "Mikaela Shiffrin", "events": [], "favorite": True})
        meta = p.VECTOR_STORE["athlete::mikaela_shiffrin"]["metadata"]
        self.assertEqual(meta["injury_risk"], "high")

    def test_athlete_metadata_no_injury_risk(self):
        p.upsert_athlete({"name": "Yuzuru Hanyu", "events": [], "favorite": True})
        meta = p.VECTOR_STORE["athlete::yuzuru_hanyu"]["metadata"]
        self.assertIsNone(meta["injury_risk"])


# ─────────────────────────────────────────────
# INDIVIDUAL UPSET DETECTION
# ─────────────────────────────────────────────
class TestIndividualUpsets(unittest.TestCase):

    def setUp(self):
        _reset()

    def _seed_full_results(self):
        """Load the same stub results the pipeline uses."""
        for event in p.discover_entities("LIVE_GAMES")["events"]:
            medalists = p.fetch_event_results(event)
            p.EVENT_RESULTS_THIS_RUN[event] = medalists

    def test_detect_upsets_fires_on_non_favorites(self):
        self._seed_full_results()
        count = p.detect_upsets()
        # Stub data produces 5 individual upsets
        self.assertEqual(count, 5)

    def test_upset_vectors_written(self):
        self._seed_full_results()
        p.detect_upsets()
        # Sara Hector (SWE) gold in downhill — not a favorite
        self.assertIn(
            "upset::women_s_downhill_alpine_skiing_sara_hector",
            p.VECTOR_STORE
        )

    def test_upset_vector_metadata(self):
        self._seed_full_results()
        p.detect_upsets()
        meta = p.VECTOR_STORE[
            "upset::women_s_downhill_alpine_skiing_sara_hector"
        ]["metadata"]
        self.assertEqual(meta["doc_type"], "upset")
        self.assertEqual(meta["country"], "SWE")
        self.assertEqual(meta["medal"], "Gold")

    def test_team_events_skipped(self):
        # Only seed a team event — should produce zero upsets
        p.EVENT_RESULTS_THIS_RUN["Women's ice hockey tournament"] = [
            {"rank": 1, "name": "Canada Women", "country": "CAN"},
            {"rank": 2, "name": "USA Women",    "country": "USA"},
            {"rank": 3, "name": "Finland Women","country": "FIN"},
        ]
        count = p.detect_upsets()
        self.assertEqual(count, 0)

    def test_favorite_winning_gold_no_upset(self):
        # Hanyu is a favorite and wins gold — no upset for him.
        # Kagiyama and Uno are not favorites → they DO get upset vectors.
        p.EVENT_RESULTS_THIS_RUN["Men's figure skating free skate"] = [
            {"rank": 1, "name": "Yuzuru Hanyu",   "country": "JPN"},
            {"rank": 2, "name": "Kagiyama Kaito", "country": "JPN"},
            {"rank": 3, "name": "Shoma Uno",      "country": "JPN"},
        ]
        p.detect_upsets()
        # Hanyu is a favorite → no upset vector
        self.assertNotIn(
            "upset::men_s_figure_skating_free_skate_yuzuru_hanyu",
            p.VECTOR_STORE
        )
        # Uno is NOT a favorite → upset vector exists
        self.assertIn(
            "upset::men_s_figure_skating_free_skate_shoma_uno",
            p.VECTOR_STORE
        )


# ─────────────────────────────────────────────
# COUNTRY UPSET DETECTION — all three signals
# ─────────────────────────────────────────────
class TestCountryUpsets(unittest.TestCase):

    def setUp(self):
        _reset()

    def _seed_full_results(self):
        for event in p.discover_entities("LIVE_GAMES")["events"]:
            p.EVENT_RESULTS_THIS_RUN[event] = p.fetch_event_results(event)

    # ── Signal 1: team_event ──
    def test_team_event_fires_when_favorite_loses(self):
        self._seed_full_results()
        p.detect_country_upsets()
        # CAN beat USA in hockey
        self.assertIn(
            "country_upset::team_event_can_women_s_ice_hockey_tournament",
            p.VECTOR_STORE
        )
        # SWE beat USA in curling
        self.assertIn(
            "country_upset::team_event_swe_men_s_curling",
            p.VECTOR_STORE
        )

    def test_team_event_metadata(self):
        self._seed_full_results()
        p.detect_country_upsets()
        meta = p.VECTOR_STORE[
            "country_upset::team_event_can_women_s_ice_hockey_tournament"
        ]["metadata"]
        self.assertEqual(meta["signal_type"], "team_event")
        self.assertEqual(meta["country"], "CAN")
        self.assertEqual(meta["favored_country"], "USA")

    def test_team_event_does_not_fire_when_favorite_wins(self):
        # Rig hockey so USA wins
        p.EVENT_RESULTS_THIS_RUN["Women's ice hockey tournament"] = [
            {"rank": 1, "name": "USA Women",     "country": "USA"},
            {"rank": 2, "name": "Canada Women",  "country": "CAN"},
            {"rank": 3, "name": "Finland Women", "country": "FIN"},
        ]
        p.detect_country_upsets()
        # No team_event vector for hockey
        hockey_vectors = [
            v for v in p.VECTOR_STORE
            if "team_event" in v and "ice_hockey" in v
        ]
        self.assertEqual(len(hockey_vectors), 0)

    # ── Signal 2: surge ──
    def test_surge_fires_for_swe(self):
        self._seed_full_results()
        p.detect_country_upsets()
        # SWE: 2 golds (downhill + curling), baseline 0, Δ+2 > threshold 1
        self.assertIn("country_upset::surge_swe", p.VECTOR_STORE)

    def test_surge_metadata(self):
        self._seed_full_results()
        p.detect_country_upsets()
        meta = p.VECTOR_STORE["country_upset::surge_swe"]["metadata"]
        self.assertEqual(meta["signal_type"], "surge")
        self.assertEqual(meta["golds_actual"], 2)
        self.assertEqual(meta["golds_baseline"], 0)
        self.assertEqual(meta["delta"], 2)

    def test_surge_does_not_fire_at_threshold(self):
        # CAN gets exactly 1 gold, baseline 0 → Δ+1 = threshold, NOT over
        self._seed_full_results()
        p.detect_country_upsets()
        self.assertNotIn("country_upset::surge_can", p.VECTOR_STORE)

    def test_surge_does_not_fire_at_baseline(self):
        # JPN gets 1 gold, baseline 1 → Δ+0
        self._seed_full_results()
        p.detect_country_upsets()
        self.assertNotIn("country_upset::surge_jpn", p.VECTOR_STORE)

    def test_surge_fires_with_custom_threshold(self):
        # Lower threshold to 0 → CAN (Δ+1) should now fire
        original = p.COUNTRY_SURGE_THRESHOLD
        p.COUNTRY_SURGE_THRESHOLD = 0
        try:
            self._seed_full_results()
            p.detect_country_upsets()
            self.assertIn("country_upset::surge_can", p.VECTOR_STORE)
        finally:
            p.COUNTRY_SURGE_THRESHOLD = original

    # ── Signal 3: shutout ──
    def test_shutout_does_not_fire_on_stub_data(self):
        # All expected countries medal in the stubs → zero shutouts
        self._seed_full_results()
        p.detect_country_upsets()
        shutout_vectors = [v for v in p.VECTOR_STORE if "shutout" in v]
        self.assertEqual(len(shutout_vectors), 0)

    def test_shutout_fires_when_expected_country_absent(self):
        # Rig downhill so CZE disappears from podium
        p.EVENT_RESULTS_THIS_RUN["Women's downhill alpine skiing"] = [
            {"rank": 1, "name": "Sara Hector",      "country": "SWE"},
            {"rank": 2, "name": "Mikaela Shiffrin", "country": "USA"},
            {"rank": 3, "name": "Some Other",       "country": "NOR"},  # CZE gone
        ]
        p.detect_country_upsets()
        self.assertIn(
            "country_upset::shutout_cze_women_s_downhill_alpine_skiing",
            p.VECTOR_STORE
        )

    def test_shutout_metadata(self):
        p.EVENT_RESULTS_THIS_RUN["Women's downhill alpine skiing"] = [
            {"rank": 1, "name": "Sara Hector",      "country": "SWE"},
            {"rank": 2, "name": "Mikaela Shiffrin", "country": "USA"},
            {"rank": 3, "name": "Some Other",       "country": "NOR"},
        ]
        p.detect_country_upsets()
        meta = p.VECTOR_STORE[
            "country_upset::shutout_cze_women_s_downhill_alpine_skiing"
        ]["metadata"]
        self.assertEqual(meta["signal_type"], "shutout")
        self.assertEqual(meta["country"], "CZE")
        self.assertEqual(meta["event"], "Women's downhill alpine skiing")

    # ── combined count ──
    def test_total_country_upsets_on_stub_data(self):
        self._seed_full_results()
        count = p.detect_country_upsets()
        # 2 team_event + 1 surge + 0 shutout = 3
        self.assertEqual(count, 3)


# ─────────────────────────────────────────────
# SUMMARY ROUTING
# ─────────────────────────────────────────────
class TestSummarizeUpdates(unittest.TestCase):

    def test_routes_all_prefixes(self):
        updates = [
            ("page::opening_ceremony",                                          "inserted"),
            ("rumor::bocelli_opening",                                          "inserted"),
            ("injury::lindsey_vonn",                                            "inserted"),
            ("athlete::nathan_chen",                                            "inserted"),
            ("event::mens_figure_skating",                                      "inserted"),
            ("upset::womens_downhill_sara_hector",                              "inserted"),
            ("country_upset::team_event_can_womens_ice_hockey_tournament",      "inserted"),
            ("country_upset::surge_swe",                                        "inserted"),
            ("country_upset::shutout_cze_womens_downhill",                      "inserted"),
        ]
        summary = p.summarize_updates(updates)
        self.assertEqual(len(summary["narratives"]),     1)
        self.assertEqual(len(summary["rumors"]),         1)
        self.assertEqual(len(summary["injuries"]),       1)
        self.assertEqual(len(summary["athletes"]),       1)
        self.assertEqual(len(summary["events"]),         1)
        self.assertEqual(len(summary["upsets"]),         1)
        self.assertEqual(len(summary["country_upsets"]), 3)

    def test_country_upset_not_routed_to_upset(self):
        # This is the prefix-ordering bug that was fixed — make sure it stays fixed
        updates = [("country_upset::surge_swe", "inserted")]
        summary = p.summarize_updates(updates)
        self.assertEqual(len(summary["country_upsets"]), 1)
        self.assertEqual(len(summary["upsets"]),         0)


# ─────────────────────────────────────────────
# FULL PIPELINE RUNS PER MODE
# ─────────────────────────────────────────────
class TestMainMode(unittest.TestCase):

    def setUp(self):
        _reset()

    def _run_in_mode(self, mode):
        """Monkey-patch resolve_mode, run main(), return vector count."""
        original = p.resolve_mode
        p.resolve_mode = lambda now=None: mode
        try:
            p.main()
        finally:
            p.resolve_mode = original
        return len(p.UPDATED_VECTORS)

    def test_pre_games_no_events_no_upsets(self):
        count = self._run_in_mode("PRE_GAMES")
        # PRE_GAMES: 3 narratives + 1 rumor + 1 injury + 10 athletes = 15
        self.assertEqual(count, 15)
        # No event or upset vectors
        event_vecs   = [v for v, _ in p.UPDATED_VECTORS if v.startswith("event::")]
        upset_vecs   = [v for v, _ in p.UPDATED_VECTORS if v.startswith("upset::")]
        country_vecs = [v for v, _ in p.UPDATED_VECTORS if v.startswith("country_upset::")]
        self.assertEqual(len(event_vecs),   0)
        self.assertEqual(len(upset_vecs),   0)
        self.assertEqual(len(country_vecs), 0)

    def test_live_games_full_pipeline(self):
        count = self._run_in_mode("LIVE_GAMES")
        # 3 narratives + 1 rumor + 1 injury + 6 events + 10 athletes
        # + 5 individual upsets + 3 country upsets = 29
        self.assertEqual(count, 29)

    def test_live_games_has_all_vector_types(self):
        self._run_in_mode("LIVE_GAMES")
        prefixes = {v.split("::")[0] for v, _ in p.UPDATED_VECTORS}
        expected = {"page", "rumor", "injury", "event", "athlete", "upset", "country_upset"}
        self.assertEqual(prefixes, expected)

    def test_dormant_writes_nothing(self):
        count = self._run_in_mode("DORMANT")
        self.assertEqual(count, 0)
        self.assertEqual(len(p.VECTOR_STORE), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
