from radar.dedupe import (clean_title, dedupe_items, event_fingerprint, hamming, is_near_duplicate,
                          item_id, jaccard, match_events, normalize_url, simhash, title_similarity,
                          tokenize)

from conftest import make_item


def test_normalize_url_strips_tracking_and_fragment():
    a = normalize_url("HTTPS://Example.com/a/b/?utm_source=x&id=3#frag")
    assert a == "https://example.com/a/b?id=3"


def test_google_news_url_drops_query():
    u = normalize_url("https://news.google.com/rss/articles/CBMiABC?oc=5")
    assert u == "https://news.google.com/rss/articles/CBMiABC"


def test_item_id_stable_across_tracking_params():
    assert item_id("https://a.com/x?utm_source=1") == item_id("https://a.com/x")


def test_clean_title_strips_publisher():
    assert clean_title("OJK tightens rules - Jakarta Post", "Jakarta Post") == "OJK tightens rules"
    assert clean_title("OJK tightens digital lending rules for fintech - Reuters") == \
        "OJK tightens digital lending rules for fintech"
    assert clean_title("Short - X") == "Short - X"  # 太短不误伤


def test_tokenize_removes_stopwords_and_plurals():
    assert tokenize("The lenders are in trouble") == ["lender", "trouble"]


def test_jaccard():
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3
    assert jaccard([], ["a"]) == 0.0


def test_simhash_similar_closer_than_different():
    a = tokenize("OJK tightens rules on digital lending platforms across Indonesia")
    b = tokenize("OJK tightens rules on digital lending platforms in Indonesia")
    c = tokenize("Grab raises funding round for Southeast Asia expansion plan")
    assert hamming(simhash(a), simhash(b)) < hamming(simhash(a), simhash(c))


def test_near_duplicate_rewording():
    a = "OJK tightens digital lending rules for fintech lenders"
    b = "OJK tightens rules on digital lending for fintech lenders"
    assert is_near_duplicate(a, b)


def test_not_duplicate_different_stories():
    a = "OJK tightens digital lending rules for fintech lenders"
    b = "Kredivo secures new debt funding from investors"
    assert not is_near_duplicate(a, b)


def test_same_entity_different_events_not_merged():
    a = "BSP issues BNPL guidelines for banks"
    b = "BSP fines bank over data breach"
    assert not is_near_duplicate(a, b)


def test_dedupe_items_keeps_highest_tier_in_batch(cfg):
    low = make_item("OJK tightens digital lending rules for fintech lenders", tier=3, hours_ago=5)
    high = make_item("OJK tightens rules on digital lending for fintech lenders", tier=1, hours_ago=1)
    other = make_item("Kredivo secures new debt funding from investors", tier=3)
    kept, dropped = dedupe_items([low, high, other], [], cfg)
    assert {k.id for k in kept} == {high.id, other.id}
    assert dropped == [(low, high.id)]


def test_dedupe_items_against_known(cfg):
    it = make_item("OJK tightens digital lending rules for fintech lenders")
    kept, dropped = dedupe_items(
        [it], [("old1", "OJK tightens rules on digital lending for fintech lenders")], cfg)
    assert kept == [] and dropped == [(it, "old1")]


def test_match_events_ranks_by_similarity():
    events = {
        "e1": ["OJK tightens digital lending rules for fintech lenders"],
        "e2": ["Kredivo secures new debt funding"],
    }
    res = match_events("OJK sets deadline for digital lending rules compliance", events, 0.3)
    assert res and res[0][0] == "e1"
    assert all(eid != "e2" for eid, _ in res)


def test_title_similarity_bounds():
    assert title_similarity("ojk lending rules", "ojk lending rules") == 1.0
    assert 0 <= title_similarity("OJK rules", "Grab funding") <= 1


def test_event_fingerprint_stable_and_country_sensitive():
    f1 = event_fingerprint("ID", "OJK tightens digital lending rules")
    assert f1 == event_fingerprint("ID", "digital lending rules OJK tightens")
    assert f1 != event_fingerprint("PH", "OJK tightens digital lending rules")
