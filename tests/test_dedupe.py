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


def test_match_events_recalls_short_followup_of_long_title():
    """回归：Jaccard 只有 0.23 的后续报道也必须进入候选（真实数据里曾被漏掉）。"""
    events = {"grab": ["Grab’s acquisition of Atome signals need for wider ecosystem play for BNPL players"]}
    assert title_similarity("Grab completes Atome acquisition after regulator nod", events["grab"][0]) < 0.3
    res = match_events("Grab completes Atome acquisition after regulator nod", events, 0.3)
    assert [eid for eid, _ in res] == ["grab"]


def test_match_events_needs_two_shared_tokens():
    events = {"e": ["OJK fines lender over data breach"]}
    assert match_events("OJK", events, 0.3) == []
    assert match_events("Weather in Jakarta today OJK", events, 0.3) == []  # 只共享 1 个词


def test_title_similarity_bounds():
    assert title_similarity("ojk lending rules", "ojk lending rules") == 1.0
    assert 0 <= title_similarity("OJK rules", "Grab funding") <= 1


def test_event_fingerprint_stable_and_country_sensitive():
    f1 = event_fingerprint("ID", "OJK tightens digital lending rules")
    assert f1 == event_fingerprint("ID", "digital lending rules OJK tightens")
    assert f1 != event_fingerprint("PH", "OJK tightens digital lending rules")


def test_date_only_difference_is_not_duplicate():
    """回归：只差日期的模板标题（汇率播报 / 月度新闻稿）曾被误并成一条。"""
    from radar.dedupe import dates_in
    a = "LOOK: Foreign Exchange Rate as of September 17, 2026 Source: BSP"
    b = "LOOK: Foreign Exchange Rate as of September 11, 2026 Source: BSP"
    assert not is_near_duplicate(a, b)
    assert not is_near_duplicate("OJK Press Release: April 2026 Board of Commissioners Monthly Meeting",
                                 "OJK Press Release: May 2026 Board of Commissioners Monthly Meeting")
    assert dates_in(a) == {("sep", "17", "2026")}
    assert dates_in("OJK Press Release: April 2026 Board Meeting") == {("apr", "", "2026")}


def test_month_like_words_are_not_dates():
    """回归：market / summary / decision / may（情态动词）里的月份缩写不能被当成日期。"""
    from radar.dedupe import dates_in
    assert dates_in("Market summary: decision on lending rules may follow") == set()
    assert dates_in("Grab market share and Decision Bank") == set()
    # 因此与不含日期的标题仍可正常合并
    assert is_near_duplicate("Grab acquires majority stake in Atome Financial market entry",
                             "Grab acquires majority stake in Atome Financial for market entry")


def test_same_date_or_no_date_still_merges():
    a = "BSP Foreign Exchange Rate as of September 17, 2026 published"
    assert is_near_duplicate(a, "BSP foreign exchange rate as of September 17, 2026 published today")
    # 只有一边带日期时不阻止合并
    assert is_near_duplicate("OJK tightens digital lending rules for fintech lenders",
                             "OJK tightens rules on digital lending for fintech lenders in September")
