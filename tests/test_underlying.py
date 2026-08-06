from data.underlying import resolve_yahoo_ticker


def test_resolves_gold_by_symbol():
    assert resolve_yahoo_ticker("GO10OZ", "Gold 10oz") == ("Gold", "GC=F")


def test_resolves_palladium_before_generic_platinum_keyword():
    assert resolve_yahoo_ticker("PALDIUM100-SE26", "PALLADIUM100") == ("Palladium", "PA=F")


def test_resolves_crude_oil():
    assert resolve_yahoo_ticker("CL100BBL", "Crude Oil 100bbl") == ("Crude Oil", "CL=F")


def test_resolves_nasdaq_index():
    assert resolve_yahoo_ticker("NSDQ100-SE26", "NSDQ100") == ("Nasdaq 100", "^IXIC")


def test_resolves_maize_to_corn():
    assert resolve_yahoo_ticker("MAIZELD-AU26", "Yellow Maize Long Dated Futures Contract") == (
        "Corn",
        "ZC=F",
    )


def test_short_keyword_does_not_false_match_inside_word():
    # "DJ" must not match just because it's a substring of another word.
    assert resolve_yahoo_ticker("ADJUSTABLE99", "Adjustable rate something") is None


def test_returns_none_for_unrecognized_symbol():
    assert resolve_yahoo_ticker("XYZ999", "Some unrelated instrument") is None
